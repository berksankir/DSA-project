# From Diplomacy Discourse to On the Ground Events

 Project Proposal:
   
I ask this question: Do weekly changes in diplomatic discourse from one country towards another country help anticipate protest/tension events in the second country over the next 1–4 weeks? 
I will create simple weekly signals from diplomatic texts (tone/stance/frames at the dyad level), match them with GDELT protest/tension indicators, and evaluate basic early warning value.
______________________________________
Country dyads and scope:

In this project, I will focus on diplomatic discourse and on-ground events in the context of the Russia–Ukraine war. Our core dyads are:

- United States – Russia (US–RU)
- United States – Ukraine (US–UA)
- Russia – Ukraine (RU–UA)

The full data pipeline (diplomatic text processing, dyad–week feature construction, and GDELT event matching) is implemented for these three dyads. 

As a possible extension (if time permits), the same pipeline can be applied to:
- United Kingdom – Russia (UK–RU)
- United Kingdom – Ukraine (UK–UA)
________________________________________
 Data To Be Used:

*	GlobalDiplomacyNet (GDN): Public diplomatic texts/relations across countries and years.

*	GDELT Events (v2): Global event records; we will use protest (root 14) and related tension/violence families (13, 15–20).

*	Lowy Global Diplomacy Index: Yearly diplomatic network/context measures.

All sources are public and will be credited and their licenses will be respected.

You can access all data that is used via this Google Drive link:
https://drive.google.com/drive/folders/1hL3xYYd8GgNYmPt2hlnrr5WSckcq8hQU?usp=sharing
________________________________________
 Plan to Collect The Data:

*	Gathering GDELT events for the study period and prepare weekly country outcomes.

*	Gathering GDN texts and prepare weekly dyad summaries (i→j).

*	Optionally adding Lowy context at the country-year level.

*	Combining everything into a single weekly dyad dataset for analysis.
________________________________________
 Hypothesis Tests

I test whether diplomatic discourse features (lexicon-based ratios/scores) are associated with next-week on-the-ground events in a dyad–week panel (2019–2024). Outcomes are next-week event counts and “any event” indicators derived from counts.

* H1 — Period Difference (Pre-war vs War)

**Periods:** Pre-war (2019–2021) vs War (2022–2024).  
**Outcomes:** `military_next_week`, `protest_next_week`, and binaries `1[count>0]`.  
**H0:** No difference between periods.  
**Tests:** Chi-square (binary), Mann–Whitney U (counts).  
**Visuals:** Period boxplots (counts) + period bar charts (event rates).

* H2 — Discourse → Next-week Military Events

**Outcome:** `military_next_week` (and `1[military_next_week>0]` as robustness).  
**Predictors:** selected discourse features (ratios/scores).  
**H0:** Discourse features do not predict next-week military events.  
**Tests/Models:** Negative Binomial regression (counts); Logistic regression (binary robustness).  
**Visuals:** Coefficient plot + predicted effect (or quantile comparison) plots.

* H3 — Discourse → Next-week Protest Events

**Outcome:** `protest_next_week` (and `1[protest_next_week>0]` as robustness).  
**Predictors:** selected discourse features (ratios/scores).  
**H0:** Discourse features do not predict next-week protest events.  
**Tests/Models:** Negative Binomial regression (counts); Logistic regression (binary robustness).  
**Visuals:** Coefficient plot + predicted effect (or quantile comparison) plots.

