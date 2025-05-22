import time
import asyncio
import numpy as np
from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

CONTENT_ARCHITECTURE_PROMPT = """As an expert content strategist, analyze the query and retrieved information to create an optimal content architecture plan.

Query: {query}
Sub-queries: {sub_queries}
Available Information Summary: {info_summary}

Based on the query type and available information, design a structured content architecture that includes:

1. Content Type Classification: Identify if this is:
   - Comparative analysis
   - Causal relationship analysis  
   - Historical development review
   - Technical explanation
   - Policy analysis
   - Other (specify)

2. Logical Structure Plan: Design the optimal organization pattern:
   - Sequential (chronological/step-by-step)
   - Categorical (thematic grouping)
   - Hierarchical (general to specific)
   - Comparative (side-by-side analysis)
   - Problem-solution framework
   - Other (specify)

3. Content Outline: Create a detailed content outline with:
   - Main sections and their logical flow
   - Key points for each section
   - Evidence allocation strategy
   - Transition strategies between sections

4. Quality Standards: Define specific quality criteria for this content type:
   - Required depth of analysis
   - Evidence standards
   - Critical thinking requirements
   - Professional terminology usage

Return your response in the following JSON format:
{{
    "content_type": "string",
    "structure_pattern": "string", 
    "outline": [
        {{
            "section": "string",
            "key_points": ["string"],
            "evidence_strategy": "string"
        }}
    ],
    "quality_criteria": {{
        "analysis_depth": "string",
        "evidence_requirements": "string",
        "critical_thinking": "string"
    }}
}}
"""

SUB_QUERY_PROMPT = """In order to answer this question more comprehensively, please break down the original question into at most four sub-questions. 
I hope you will not simply split the original question directly, but after summarizing the original question, get at most four sub-questions related to the direction of the original question.
These sub-questions can cover the core elements of the original question very well, and each sub-question has the value of independent exploration.Also, sub-questions should be general and not too detailed.
Return as list of str.If this is a very simple question and no decomposition is necessary, then keep the only one original question in the python code list.

Original Question: {original_query}


<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning and what are its core principles?",
    "How does deep learning compare to other machine learning approaches?",
    "What are the main types of deep learning models and their general applications?",
    "What are the key advantages and challenges of deep learning?",
]
</EXAMPLE>

Provide your response in a python code list of str format:
"""

RERANK_PROMPT = """Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful in answering the any of the questions?
"""


REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

If the original query is to write a report, then you prefer to generate some further queries, instead return an empty list.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}

Respond exclusively in valid List of str format without any other text."""


SUMMARY_PROMPT = """You are an expert academic writer tasked with creating a sophisticated synthesis based on the provided content architecture and retrieved information.

Content Architecture Plan:{architecture_plan}

Writing Context:
- Original Query: {question}
- Sub-queries Explored: {mini_questions}
- Content Type: {content_type}
- Structure Pattern: {structure_pattern}

Quality Standards for This Content:
- Analysis Depth: {analysis_depth}
- Evidence Requirements: {evidence_requirements}
- Critical Thinking: {critical_thinking}

Source Materials:
{mini_chunk_str}

Detailed Writing Instructions:
1. Role-Specific Expertise: 
Write as a leading expert in the relevant field with deep knowledge and analytical capabilities.

2. Structure Implementation:
    - Follow the provided outline structure precisely
   - Ensure logical flow between sections as planned
   - Implement the specified evidence allocation strategy

3. Analysis Requirements:
   - Go beyond information summarization to provide deep analytical insights  
   - Identify patterns, relationships, and implications not explicitly stated in sources
   - Present critical evaluation of different perspectives where applicable
   - Draw meaningful conclusions based on evidence synthesis

4.Evidence Integration：
   - Seamlessly weave evidence into argumentative flow
   - Prioritize high-quality, credible sources
   - Address potential counterarguments or limitations
   - Maintain clear traceability between claims and evidence
   
5.Language and Style Excellence：
   - Use precise, field-appropriate terminology consistently
   - Employ sophisticated academic discourse
   - Create smooth transitions that guide reader understanding
   - Vary sentence structure for engaging readability
   
6.Critical Thinking Demonstration：
   - Question assumptions and examine multiple perspectives
   - Identify cause-effect relationships and their implications
   - Highlight areas of uncertainty or ongoing debate
   - Suggest future research directions or practical applications

Output Format Requirements：

\\documentclass {{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{booktabs}}
\\usepackage{{parskip}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{1em}}

\\begin{{document}}

\\title{{[Generate an insightful title that captures the analytical focus]}}
\\author{{AI Research Synthesis}}
\\date{{\\today}}
\\maketitle

[Begin with your sophisticated academic analysis following the architecture plan...]

\\end{{document}}

Remember: 
This should read as an authoritative, insightful analysis that demonstrates expertise beyond simple information compilation. Focus on providing value through analytical depth and critical evaluation.
"""

QUERY_TYPE_ANALYSIS_PROMPT = """Analyze the following query to determine its type and complexity characteristics:

Query: {query}

Classify this query across the following dimensions:

1. Primary Intent:
   - Information seeking (what/who/when/where)
   - Analysis seeking (why/how/implications)
   - Comparison seeking (differences/similarities)
   - Synthesis seeking (comprehensive overview)
   - Problem-solving seeking (solutions/recommendations)

2. Domain Complexity:
   - Single domain focus
   - Multi-domain integration required
   - Cross-disciplinary analysis needed

3. Temporal Dimension:
   - Historical focus
   - Current state analysis
   - Future-oriented/predictive
   - Longitudinal development

4. Cognitive Demand:
   - Factual recall
   - Conceptual understanding
   - Analytical reasoning
   - Creative synthesis
   - Critical evaluation

Return as JSON:
{{
    "primary_intent": "string",
    "domain_complexity": "string", 
    "temporal_dimension": "string",
    "cognitive_demand": "string",
    "complexity_score": 1-10,
    "recommended_approach": "string"
}}
"""


@describe_class(
    "This agent is suitable for handling open-ended, general, or summary-style questions, especially those requiring content generation such as writing reports, surveys, or articles. "
    "It is also able to retrieve and integrate information from large text collections to provide synthesized responses."
)
class DeepSearch(RAGAgent):
    """
    Deep Search agent implementation for comprehensive information retrieval.

    This agent performs a thorough search through the knowledge base, analyzing
    multiple aspects of the query to provide comprehensive and detailed answers.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 3,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        """
        Initialize the DeepSearch agent.

        Args:
            llm: The language model to use for generating answers.
            embedding_model: The embedding model to use for query embedding.
            vector_db: The vector database to search for relevant documents.
            max_iter: The maximum number of iterations for the search process.
            route_collection: Whether to use a collection router for search.
            text_window_splitter: Whether to use text_window splitter.
            **kwargs: Additional keyword arguments for customization.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        self.collection_router = CollectionRouter(
            llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.text_window_splitter = text_window_splitter

    def _analyze_query_characteristics(self, query: str) -> Tuple[dict, int]:
        """Analyze query characteristics to provide a basis for content architecture planning"""
        chat_response = self.llm.chat(
            messages=[{"role": "user", "content": QUERY_TYPE_ANALYSIS_PROMPT.format(query=query)}]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def _plan_content_architecture(self, query: str, sub_queries: List[str], chunks: List[RetrievalResult]) -> Tuple[
        dict, int]:
        """Intelligent content architecture planning"""
        info_summary = self._generate_info_summary([chunk.text for chunk in chunks[:10]])  # 使用前10个chunk生成摘要

        chat_response = self.llm.chat(
            messages=[{
                "role": "user",
                "content": CONTENT_ARCHITECTURE_PROMPT.format(
                    query=query,
                    sub_queries=sub_queries,
                    info_summary=info_summary
                )
            }]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def _generate_info_summary(self, chunk_texts: List[str]) -> str:
        """Generate information summaries for architecture planning"""
        if not chunk_texts:
            return "No information available."

        sample_texts = chunk_texts[:5]
        combined_text = " ".join(sample_texts)

        if len(combined_text) > 1000:
            combined_text = combined_text[:1000] + "..."

        return f"Available information covers: {combined_text}"

    def _apply_role_based_prompting(self, base_prompt: str, query_characteristics: dict) -> str:
        """Apply role-based prompts based on query characteristics"""
        domain_complexity = query_characteristics.get("domain_complexity", "single")
        cognitive_demand = query_characteristics.get("cognitive_demand", "factual")

        if "multi-domain" in domain_complexity.lower():
            role_prefix = "As an interdisciplinary research expert with deep knowledge across multiple fields, "
        elif "cross-disciplinary" in domain_complexity.lower():
            role_prefix = "As a cross-disciplinary scholar specializing in connecting insights across different domains, "
        elif cognitive_demand in ["analytical reasoning", "critical evaluation"]:
            role_prefix = "As a senior analytical researcher with expertise in critical evaluation and complex reasoning, "
        elif cognitive_demand == "creative synthesis":
            role_prefix = "As an expert synthesis researcher skilled in creative integration of diverse information sources, "
        else:
            role_prefix = "As a domain expert with comprehensive knowledge and analytical capabilities, "

        return role_prefix + base_prompt

    def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
        chat_response = self.llm.chat(
            messages=[
                {"role": "user", "content": SUB_QUERY_PROMPT.format(original_query=original_query)}
            ]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    async def _search_chunks_from_vectordb(self, query: str, sub_queries: List[str]):
        consume_tokens = 0
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(
                query=query, dim=self.embedding_model.dimension
            )
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route

        all_retrieved_results = []
        if not query.strip():
            log.color_print(f"<search> Skipping search for empty or whitespace query. </search>\n")
            return all_retrieved_results, consume_tokens

        query_vector = self.embedding_model.embed_query(query)
        for collection in selected_collections:
            log.color_print(f"<search> Search [{query}] in [{collection}]...  </search>\n")
            retrieved_results = self.vector_db.search_data(
                collection=collection, vector=query_vector
            )
            if not retrieved_results or len(retrieved_results) == 0:
                log.color_print(
                    f"<search> No relevant document chunks found in '{collection}'! </search>\n"
                )
                continue
            accepted_chunk_num = 0
            references = set()
            for retrieved_result in retrieved_results:
                rerank_queries = [q for q in ([query] + sub_queries) if q and q.strip()]
                if not rerank_queries:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)
                    continue

                chat_response = self.llm.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": RERANK_PROMPT.format(
                                query=[query] + sub_queries,
                                retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",
                            ),
                        }
                    ]
                )
                consume_tokens += chat_response.total_tokens
                response_content = chat_response.content.strip()
                # strip the reasoning text if exists
                if "<think>" in response_content and "</think>" in response_content:
                    end_of_think = response_content.find("</think>") + len("</think>")
                    response_content = response_content[end_of_think:].strip()
                if "YES" in response_content and "NO" not in response_content:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)
            if accepted_chunk_num > 0:
                log.color_print(
                    f"<search> Accept {accepted_chunk_num} document chunk(s) from references: {list(references)} </search>\n"
                )
            else:
                log.color_print(
                    f"<search> No document chunk accepted from '{collection}'! </search>\n"
                )
        return all_retrieved_results, consume_tokens

    def _generate_gap_queries(
        self, original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]
    ) -> Tuple[List[str], int]:
        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts([chunk.text for chunk in all_chunks])
            if len(all_chunks) > 0
            else "NO RELATED CHUNKS FOUND.",
        )
        chat_response = self.llm.chat([{"role": "user", "content": reflect_prompt}])
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve relevant documents from the knowledge base for the given query.

        This method performs a deep search through the vector database to find
        the most relevant documents for answering the query.

        Args:
            original_query (str): The query to search for.
            **kwargs: Additional keyword arguments for customizing the retrieval.

        Returns:
            Tuple[List[RetrievalResult], int, dict]: A tuple containing:
                - A list of retrieved document results
                - The token usage for the retrieval operation
                - Additional information about the retrieval process
        """
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    async def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        max_iter = kwargs.pop("max_iter", self.max_iter)
        min_info_gain_threshold = kwargs.pop("min_info_gain", 0.1)

        ### SUB QUERIES ###
        log.color_print(f"<query> {original_query} </query>\n")
        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        sub_queries, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token

        if not sub_queries:
            log.color_print("No sub queries were generated by the LLM. Using original query for search.")
            if original_query and original_query.strip():
                sub_queries = [original_query]
            else:
                log.color_print("Original query is also empty. Exiting retrieval.")
                return all_search_res, total_tokens, {"all_sub_queries": [], "iterations_performed": 0}
        else:
            log.color_print(
                f"<think> Break down the original query into new sub queries: {sub_queries}</think>\n"
            )

        all_sub_queries.extend(sub_queries)
        sub_gap_queries = [q for q in sub_queries if q and q.strip()]

        previous_iteration_results = []
        iterations_performed = 0

        for iter in range(max_iter):
            iterations_performed = iter+ 1
            log.color_print(f">> Iteration: {iterations_performed}\n")
            current_iteration_results = []

            if not sub_gap_queries:
                log.color_print(
                    f"<think> No valid sub-gap queries for iteration {iterations_performed}. Skipping search phase. </think>\n")
            else:
                search_tasks = [
                    self._search_chunks_from_vectordb(query, sub_gap_queries)
                    for query in sub_gap_queries
                ]
                search_results = await asyncio.gather(*search_tasks)

                for result in search_results:
                    search_res, consumed_token = result
                    total_tokens += consumed_token
                    current_iteration_results.extend(search_res)

            current_iteration_results = deduplicate_results(current_iteration_results)

            info_gain = self._calculate_information_gain(
                previous_iteration_results if previous_iteration_results else all_search_res,
                current_iteration_results
            )

            log.color_print(f"<think> Information gain from iteration {iter + 1}: {info_gain:.4f} </think>\n")

            all_search_res.extend(current_iteration_results)
            all_search_res = deduplicate_results(all_search_res)

            previous_iteration_results = current_iteration_results

            if iter > 0 and info_gain < min_info_gain_threshold:
                log.color_print(
                    f"<think> Information gain below threshold ({info_gain:.4f} < {min_info_gain_threshold}). Stopping iterations. </think>\n")
                break

            if iterations_performed == max_iter:
                log.color_print("<think> Reached maximum iterations. Stopping. </think>\n")
                break

            log.color_print("<think> Reflecting on the search results... </think>\n")
            gap_queries_list, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token

            sub_gap_queries = [q for q in gap_queries_list if q and q.strip()]

            if not sub_gap_queries or len(sub_gap_queries) == 0:
                log.color_print("<think> No new search queries were generated. Exiting. </think>\n")
                break
            else:
                log.color_print(
                    f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n"
                )
                all_sub_queries.extend(sub_gap_queries)
                all_sub_queries = list(dict.fromkeys(all_sub_queries))

        additional_info = {
            "all_sub_queries": all_sub_queries,
            "iterations_performed": iterations_performed,
        }

        return all_search_res, total_tokens, additional_info

    def _calculate_information_gain(self, previous_results: List[RetrievalResult], new_results: List[RetrievalResult]) -> float:
        if not previous_results:
            return 1.0

        if not new_results:
            return 0.0

        prev_text = " ".join([r.text for r in previous_results])
        prev_embedding = self.embedding_model.embed_query(prev_text)

        new_text = " ".join([r.text for r in new_results])
        new_embedding = self.embedding_model.embed_query(new_text)

        similarity = np.dot(prev_embedding, new_embedding)

        information_gain = 1.0 - similarity

        volume_factor = min(len(new_results) / max(len(previous_results), 1), 1.0)

        return information_gain * volume_factor

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and generate an answer based on retrieved documents.

        This method retrieves relevant documents and uses the language model
        to generate a comprehensive answer to the query.

        Args:
            query (str): The query to answer.
            **kwargs: Additional keyword arguments for customizing the query process.

        Returns:
            Tuple[str, List[RetrievalResult], int]: A tuple containing:
                - The generated answer
                - A list of retrieved document results
                - The total token usage
        """
        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
        if not all_retrieved_results or len(all_retrieved_results) == 0:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval
        all_sub_queries = additional_info["all_sub_queries"]
        total_tokens = n_token_retrieval

        query_characteristics, tokens_analysis = self._analyze_query_characteristics(query)
        total_tokens += tokens_analysis
        log.color_print(f"<think> Query characteristics: {query_characteristics} </think>\n")

        architecture_plan, tokens_architecture = self._plan_content_architecture(
            query, all_sub_queries, all_retrieved_results
        )
        total_tokens += tokens_architecture
        log.color_print(
            f"<think> Content architecture planned: {architecture_plan.get('structure_pattern', 'undefined')} </think>\n")

        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)

        log.color_print(
            f"<think> Summarize answer from all {len(all_retrieved_results)} retrieved chunks with session context... </think>\n"
        )

        summary_prompt = SUMMARY_PROMPT.format(
            architecture_plan=architecture_plan,
            question=query,
            mini_questions=all_sub_queries,
            content_type=architecture_plan.get("content_type", "analysis"),
            structure_pattern=architecture_plan.get("structure_pattern", "hierarchical"),
            analysis_depth=architecture_plan.get("quality_criteria", {}).get("analysis_depth", "comprehensive"),
            evidence_requirements=architecture_plan.get("quality_criteria", {}).get("evidence_requirements", "multiple sources"),
            critical_thinking=architecture_plan.get("quality_criteria", {}).get("critical_thinking", "required"),
            mini_chunk_str=self._format_chunk_texts(chunk_texts),
        )

        enhanced_summary_prompt = self._apply_role_based_prompting(summary_prompt, query_characteristics)

        chat_response = self.llm.chat([{"role": "user", "content": enhanced_summary_prompt}])
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(chat_response.content)

        return (
            chat_response.content,
            all_retrieved_results,
            total_tokens + chat_response.total_tokens,
        )

    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
        return chunk_str
