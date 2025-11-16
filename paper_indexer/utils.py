import asyncio
from typing import Any

import httpx


async def retry_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> httpx.Response:
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2**attempt)
                await asyncio.sleep(wait_time)
            else:
                raise last_exception
    raise last_exception or httpx.HTTPError("Request failed")
