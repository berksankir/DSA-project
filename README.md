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
________________________________________
 Plan to Collect The Data:

*	Gathering GDELT events for the study period and prepare weekly country outcomes.

*	Gathering GDN texts and prepare weekly dyad summaries (i→j).

*	Optionally adding Lowy context at the country-year level.

*	Combining everything into a single weekly dyad dataset for analysis.
