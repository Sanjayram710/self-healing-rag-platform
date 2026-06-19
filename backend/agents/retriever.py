from backend.vectorstore.chroma import get_vectorstore
import logging


def retrieve_documents(query, collection_id=None, user_id=None):
    db = get_vectorstore()
    
    search_kwargs = {}
    
    filters = []
    if user_id:
        filters.append({"user_id": user_id})
    if collection_id and collection_id != "all":
        filters.append({"collection_id": collection_id})
        
    if len(filters) == 1:
        search_kwargs["filter"] = filters[0]
    elif len(filters) > 1:
        search_kwargs["filter"] = {"$and": filters}

    try:
        # Retrieve top k documents with relevance scores and log details
        k = max(5, 8)  # Ensure at least 5 documents are retrieved
        scored_docs = db.similarity_search_with_relevance_scores(query, k=k, **search_kwargs)
        logger = logging.getLogger('performance')
        logger.info(f"[RETRIEVE] Retrieved {len(scored_docs)} chunks (similarity search) for query: '{query}'")
        # Log top chunk preview and source information
        if scored_docs:
            top_doc, top_score = scored_docs[0]
            source = top_doc.metadata.get('source', 'unknown')
            page = top_doc.metadata.get('page', 'N/A')
            preview = top_doc.page_content[:200].replace('\n', ' ')
            logger.info(f"[RETRIEVE] Top chunk source: {source}, page: {page}, score: {top_score:.3f}, preview: {preview}")
        filtered_docs = [
            doc for doc, score in scored_docs
            if float(score or 0) >= 0.35
        ]
        if filtered_docs:
            return filtered_docs[:4]
    except Exception:
        pass

    try:
        # Use MMR to get diverse results, ensuring at least 5 documents
        mmr_k = max(5, 4)
        docs = db.max_marginal_relevance_search(query, k=mmr_k, fetch_k=12, **search_kwargs)
        logger.info(f"[RETRIEVE] Retrieved {len(docs)} chunks (MMR) for query: '{query}'")
        if docs:
            return docs
    except Exception:
        pass

    # Fallback similarity search with at least 5 results
    fallback_k = max(5, 3)
    return db.similarity_search(query, k=fallback_k, **search_kwargs)
