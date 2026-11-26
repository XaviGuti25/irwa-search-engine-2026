import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

NO_PRODUCTS = "There are no good products that fit the request based on the retrieved results."


class RAGGenerator:
    """Summary of the results with an LLM (Groq). mode='baseline' is the original template, mode='improved' is ours."""

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Identify the single best product that matches the user's request.
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        3. If there is another product that could also work, mention it briefly as an alternative.
        4. If no product is a good fit, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """

    # Improvements 1 and 2: products come with their metadata and the rules are stricter
    IMPROVED_PROMPT = """You are a shopping assistant for a fashion e-commerce site. Prices are in Indian rupees (Rs).

Rules:
- Recommend ONLY products from the list below, and always write their PID. Never invent products or attributes.
- Check every requirement in the user request (gender, colour, type of product, material, price limit...). A product that breaks one of them is not a good fit.
- Prefer products that are in stock, with a higher rating and a better price. Say if the best option is out of stock.
- {match_note}
- If no product is a good fit, answer ONLY: "{no_products}"

Retrieved products (ordered by our search engine, best first):
{retrieved_results}

User request: {user_query}

Answer in this format, in at most 120 words:
- Best product: [PID] [title]
- Why: [one or two sentences using the attributes above]
- Alternative: [PID] [title] and why (only if there is a good one)
"""

    AND_NOTE = "All the products contain every word of the request."
    OR_NOTE = "No product contains all the words of the request, so these are only partial matches. Be strict."

    @staticmethod
    def format_product(res):
        """One line per product with the fields the LLM needs to compare them."""
        stock = "OUT OF STOCK" if res.out_of_stock else "in stock"
        rating = res.average_rating if res.average_rating is not None else "no rating"
        desc = (res.description or "")[:200].replace("\n", " ")
        num = lambda x: "?" if x is None else f"{x:g}"
        return (f"- PID: {res.pid} | {res.title} | brand: {res.brand} | price: Rs {num(res.selling_price)} "
                f"(was Rs {num(res.actual_price)}, {num(res.discount or 0)}% off) | rating: {rating} | {stock} | {desc}")

    def build_prompt(self, user_query, retrieved_results, mode="improved", match_mode="and", top_N=20):
        """Return the prompt for the chosen mode."""
        if mode == "baseline":
            formatted = "\n".join(f"- PID: {r.pid}, Title: {r.title}" for r in retrieved_results[:top_N])
            return self.PROMPT_TEMPLATE.format(retrieved_results=formatted, user_query=user_query)
        formatted = "\n".join(self.format_product(r) for r in retrieved_results[:10])
        note = self.OR_NOTE if match_mode == "or" else self.AND_NOTE
        return self.IMPROVED_PROMPT.format(retrieved_results=formatted, user_query=user_query,
                                           match_note=note, no_products=NO_PRODUCTS)

    def generate_response(self, user_query: str, retrieved_results: list, mode="improved", match_mode="and",
                          top_N: int = 20) -> str:
        """Generate the summary text. Returns an error message if the API is not available."""
        # Improvement 3: with no results we answer directly, without calling the LLM
        if mode == "improved" and not retrieved_results:
            return NO_PRODUCTS
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
            prompt = self.build_prompt(user_query, retrieved_results, mode, match_mode, top_N)
            extra = {"temperature": 0.2} if mode == "improved" else {}   # lower temperature = more stable answers
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model=model_name, **extra)
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return "RAG is not available. Check your credentials (.env file) or account limits."
