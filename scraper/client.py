import httpx


class RiyasewanaClient:
    BASE_URL = "https://riyasewana.com"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def get(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/139.0 Safari/537.36"
            )
        }

        with httpx.Client(
            timeout=self.timeout,
            headers=headers,
            follow_redirects=True
        ) as client:

            response = client.get(url)
            response.raise_for_status()

            return response.text
