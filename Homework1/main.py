import minsearch
from dotenv import load_dotenv
from openai import OpenAI
from gitsource import GithubRepositoryDataReader, chunk_documents
from toyaikit import tools
load_dotenv()

client = OpenAI()

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for file in files:
    doc = file.parse()
    documents.append(doc)

# Q1. How many lesson pages?
print(f"Q1 - Number of lesson pages: {len(documents)}")

#Q2. Indexing and searching
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
index.fit(documents)

query = "How does the agentic loop keep calling the model until it stops?"
results = index.search(query, num_results=5)
print(f"Q2 Answer: {results[0]['filename']}")

#Q3. RAG
def rag(query, search_index, top_k=3):
    results = search_index.search(query, num_results=top_k)
    context = "\n\n".join([r['content'] for r in results])
    prompt = f"Answer using this context:\n{context}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content, response.usage

answer, usage = rag(query, index)
print(f"Q3 Input tokens: {usage.prompt_tokens}")

#Q4. Chunking
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Q4 Number of chunks: {len(chunks)}")

chunk_dicts = []
for chunk in chunks:
    chunk_dicts.append({
        "content": chunk["content"],
        "filename": chunk["filename"]
    })

#Q5. RAG with chunking
chunk_index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
chunk_index.fit(chunk_dicts)

answer_chunked, usage_chunked = rag(query, chunk_index)

if usage_chunked.prompt_tokens > 0:
    reduction = usage.prompt_tokens / usage_chunked.prompt_tokens
else:
    reduction = 1.0

print(f"Q5 Chunked input tokens: {usage_chunked.prompt_tokens}")
print(f"Q5 Token reduction: {reduction:.1f}x fewer tokens")

#Q6. Turning it into an agent
def search_tool(query: str, top_k: int = 3) -> str:
    results = chunk_index.search(query, num_results=top_k)
    
    if not results:
        return "No relevant information found."
    
    context = []
    for i, result in enumerate(results):
        context.append(f"[{i+1}] From {result['filename']}:\n{result['content'][:500]}...")
    
    return "\n\n".join(context)

def agent_loop(question, max_iterations=5):
    
    tool_call_count = 0
    messages = [
        {"role": "system", "content": """You're a course teaching assistant. Answer the student's question using the search tool. 
Make multiple searches with different keywords before answering.

You have access to a search function. When you want to search, respond with:
SEARCH: <your query>

When you have enough information to answer, respond with:
ANSWER: <your final answer>"""},
        {"role": "user", "content": question}
    ]
    
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            temperature=0.0
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        if assistant_message.startswith("SEARCH:"):
            search_query = assistant_message.replace("SEARCH:", "").strip()
            tool_call_count += 1
            
            search_results = search_tool(search_query)
            
            messages.append({
                "role": "user", 
                "content": f"Search results for '{search_query}':\n{search_results}"
            })
            
        elif assistant_message.startswith("ANSWER:"):
            final_answer = assistant_message.replace("ANSWER:", "").strip()
            return final_answer, tool_call_count
        
        else:
            messages.append({
                "role": "user",
                "content": "Please respond with either SEARCH: <query> or ANSWER: <your answer>"
            })
    
    return "Max iterations reached", tool_call_count

question_q6 = "How does the agentic loop work and how is it different from plain RAG?"
final_answer, tool_calls = agent_loop(question_q6)

print(f"Q6 Response: {final_answer}")
print(f"Q6 Number of search tool calls: {tool_calls}")