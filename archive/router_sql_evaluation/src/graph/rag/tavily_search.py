def tavily_search(client, query,
                  search_depth="advanced",
                  include_answer=False,
                  include_images=False,
                  include_raw_content=False,
                  max_results=5,
                  topic=None):
    params = {
        "query": query,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_images": include_images,
        "include_raw_content": include_raw_content,
        "max_results": max_results,
    }
    if topic is not None:
        params["topic"] = topic

    return client.search(**params)


def tavily_answer(client, query):
    res = tavily_search(
        client=client,
        query=query,
        search_depth="advanced",
        include_answer=False,
        include_images=False,
        include_raw_content=False,
        max_results=3,
    )
    return res.get("answer") or res


def extract_contexts_from_tavily(tavily_result, max_contexts=None):
    contexts = []
    sources = []

    results = tavily_result.get("results", [])

    for item in results:
        title = item.get("title") or "No title"
        url = item.get("url") or "No URL"
        score = item.get("score")
        content = item.get("content") or item.get("raw_content") or ""

        if not content:
            continue

        sources.append(
            {
                "title": title,
                "url": url,
                "score": score,
                "content": content,
            }
        )

        header_parts = [title]
        if isinstance(score, (int, float)):
            header_parts.append(f"score={score:.4f}")
        header = " | ".join(header_parts)

        context_piece = f"""
            Source: {header}
            URL: {url}

            Content:
            {content}
        """
        contexts.append(context_piece)

    # Limit contexts if max_contexts is specified
    if max_contexts is not None:
        return contexts[:max_contexts], sources[:max_contexts]
    return contexts, sources
