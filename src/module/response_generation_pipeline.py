import json
from typing import cast

import requests
from haystack import Document

from constants import GEMINI_API_KEY, GEMINI_MODEL


class ResponseGenerationPipeline:
    def __init__(
        self,
        prompt_template: str,
        timeout: int = 60,
    ):
        self.prompt_template = prompt_template
        self.timeout = timeout

    @staticmethod
    def aggregate_context(context: list[Document]) -> str:
        """
        Combine all the retrieved contexts into a single string

        :param context: Retrieved context from Haystack Retriever
        """

        context_list = []
        for i, c in enumerate(context, start=1):
            context_list.append(f"Context {i}\n{c.content}")

        return "\n\n".join(context_list)

    def build_prompt(self, question: str, agg_context: str) -> str:
        """Insert context and question into the prompt template"""

        return self.prompt_template.replace(
            "<<retrieved_context>>", agg_context
        ).replace("<<question>>", question)

    def generate_response(
        self,
        question: str,
        context: list[Document],
    ) -> str:

        agg_context = self.aggregate_context(context)
        prompt = self.build_prompt(question, agg_context)
        return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }

        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        data = resp.json()

        return cast(str, data["candidates"][0]["content"]["parts"][0]["text"])
