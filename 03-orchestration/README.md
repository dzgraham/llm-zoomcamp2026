#LLM-Zoomcamp-2026
#Homework3 for Daniel Graham

After obtaining the API keys and exporting them to Kestra, the flows were executed and the logs-usage observed

Question 1: AI Copilot has access to current Kestra plugin documentation

Question 2: Flow 1, the non-RAG flow, produced a vague and seemingly made up answer.

Question 3: After executing Flow 4 with a summary_length of short, the Multi-lingual Agent has an output token count of 85 and an English Brevity count of 40

Question 4: After executing Flow 4 with a summary_length of long, the Multi-lingual Agent has an output token count of 180 and an English Brevity count of 52

Question 5: After executing Flow 4 with a summary_length of long and a prompt asking for 'exactly 3 sentences', the Multi-lingual Agent has an output token count of 195 and an English Brevity count of 92

Question 6: For strict compliance and deterministic results, predictability is needed. AI agents introduce variability. Traditional task-based workflows, like the ones in Kestra are repeatable and their logic is transparent and auditable.