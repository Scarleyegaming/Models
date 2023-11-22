def summarizer(text):
    import cohere
    co = cohere.Client("uFEkgXwG8XBZ3VECGYVGZDXtEvybXwOplPq0j2aU")

    response = co.summarize(
        text=text,
        model='command',
        length='long',
        extractiveness='high',
        temperature=1,
        format="paragraph"
    )

    summary = response.summary
    
    return summary

