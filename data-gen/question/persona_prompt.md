From the documents, pick ONLY the personas who would realistically need to act on this information.

DOCUMENTS
{document_content}

PERSONA IDS (use exactly these strings)
employee, employer, hr_manager, domestic_worker, job_seeker, business_owner,
contractor, legal_advisor, government_official, student, emirati, agency

RULES

- Relevance = the persona would take an action, make a decision, or resolve a problem using this content.
- Score each persona 0–10. Include ONLY those with score ≥ 7.
- Return 2–6 personas. If fewer than 2 meet the threshold, return the top 2 anyway.
- Prefer precision over recall; do NOT include broad-interest personas.
- Disambiguation:
  • employer vs hr_manager → employer if decisions/liability/costs; hr_manager if internal compliance/policy workflows.  
  • domestic_worker only if the content targets domestic-worker rights/contracts/procedures.  
  • agency only if recruitment/placement/repatriation/temporary accommodation/agency obligations are specified.  
  • legal_advisor only if disputes/appeals/interpretation/procedure nuance is central.  
  • government_official only if inspections/enforcement/penalties/forms for officials are present.
