# LLM Boilerplate-Stripping: Before/After Examples

Three counties from `reembed_source_a_llm.py` (gemma2:9b via Ollama), chosen to span a small
sparse article, a mid-size article, and a dense major-city article. Source: `source_a_llm_cleaning_progress.jsonl`.

## Autauga County, Alabama (281 → 158 chars, -44%)

**Raw Wikipedia intro:**
> County in Alabama, United States County in Alabama Autauga County is a county located in the central portion of the U.S. state of Alabama . As of the 2020 census the population was 58,805. Its county seat is Prattville . Autauga County is part of the Montgomery metropolitan area .

**LLM-cleaned:**
> Autauga County is located in the central portion of Alabama.  As of the 2020 census the population was 58,805. It is part of the Montgomery metropolitan area.

**What changed:** dropped the duplicated "County in Alabama" nav-label boilerplate and the
"Its county seat is Prattville" sentence entirely — the model appears to have judged the
seat name as generic boilerplate rather than a distinctive fact, even though the prompt says
to keep such values if stated distinctively. Everything else (location, population, metro area)
survived.

---

## Pulaski County, Arkansas (1,079 → 699 chars, -35%)

**Raw Wikipedia intro:**
> County in Arkansas, United States ‹ The template Infobox settlement is being considered for merging . › County in Arkansas Pulaski County is a county in the U.S. state of Arkansas . With a population of 399,125 as of the 2020 United States census , it is the most populous county in Arkansas. The county is included in the Little Rock – North Little Rock – Conway metropolitan area. Its county seat is Little Rock, which is also Arkansas's capital and largest city. Pulaski County is Arkansas's fifth county, formed on December 15, 1818, alongside Clark and Hempstead Counties. Pulaski County is named for Brigadier General Casimir Pulaski , a Polish -born Continental Army officer who was killed in action at the Siege of Savannah during the Revolutionary War . The county was the site of the Battle of Bayou Fourche on September 10, 1863. The Union army took control the same day and occupied Pulaski County until the end of the Civil War . The county was home to Willow Springs Water Park , one of the oldest water parks in the nation, which opened in 1928 and closed in 2013.

**LLM-cleaned:**
> Pulaski County is the most populous county in Arkansas, with a population of 399,125 as of the 2020 United States census . It was formed on December 15, 1818, alongside Clark and Hempstead Counties. The county is named for Brigadier General Casimir Pulaski , a Polish -born Continental Army officer who was killed in action at the Siege of Savannah during the Revolutionary War .  The Battle of Bayou Fourche took place in Pulaski County on September 10, 1863. The Union army took control the same day and occupied Pulaski County until the end of the Civil War . The county was home to Willow Springs Water Park , one of the oldest water parks in the nation, which opened in 1928 and closed in 2013.

**What changed:** dropped the Wikipedia template-merge notice, the "County in Arkansas" nav
label, and the "Its county seat is Little Rock, which is also Arkansas's capital" sentence
(notable loss — that's a distinctive fact, not boilerplate). Kept population, founding date,
namesake, battle history, and the water park anecdote. All retained facts are verbatim from
the source, no fabrication observed.

---

## San Francisco County, California (3,933 → 3,534 chars, -10%)

**Raw Wikipedia intro (abridged — full text is 8 sentences of demographics/rankings, a history
paragraph, an economy paragraph, and a landmarks/institutions paragraph):**
> City and county in California, US This article is about the city and county in California. For other uses, see San Francisco (disambiguation) . Consolidated city-county in California, United States San Francisco , officially the City and County of San Francisco, is the fourth-most populous city in California and the 17th-most populous in the United States , with a population of 826,079 in 2025. [...] Prior to European settlement , San Francisco was inhabited by the Yelamu Ohlone. On June 29, 1776, settlers from New Spain established the Presidio of San Francisco [...] After the war, the confluence of returning servicemen, significant immigration , liberalizing attitudes, the rise of the beatnik and hippie countercultures, the sexual revolution , opposition to U.S. involvement in the Vietnam War , and other factors led to the Summer of Love and the gay rights movement, cementing San Francisco as a center of liberal activism . [...]

**LLM-cleaned (abridged):**
> San Francisco is the fourth-most populous city in California and the 17th-most populous in the United States, with a population of 826,079 in 2025. [...]
>
> Prior to European settlement, San Francisco was inhabited by the Yelamu Ohlone. On June 29, 1776, settlers from New Spain established the Presidio of San Francisco [...]
>
> After the war, San Francisco became a center of liberal activism due to the confluence of returning servicemen, significant immigration, liberalizing attitudes, the rise of the beatnik and hippie countercultures, the sexual revolution, opposition to U.S. involvement in the Vietnam War, and other factors leading to the Summer of Love and the gay rights movement.
>
> [...]

**What changed:** dropped only the disambiguation/nav preamble ("This article is about...",
"City and county in California, US", "Consolidated city-county..."). The body content is
almost untouched — reordered into paragraph breaks and a couple of sentences rephrased for
flow (e.g. "cementing San Francisco as a center..." → "San Francisco became a center of...
due to..."), but no facts added or dropped. Makes sense: a dense, information-rich article
has very little boilerplate-to-content ratio, so the -10% reduction is almost entirely the
generic nav sentences at the top.

## Pattern across all three

- Reduction shrinks as article richness grows: 44% → 35% → 10%.
- Consistently stripped: duplicated nav/category labels ("County in X, United States" ×2),
  disambiguation notices, Wikipedia maintenance templates ("infobox... being considered for
  merging").
- Inconsistently stripped: the "county seat is X" sentence was removed in both Alabama and
  Arkansas examples even when it carried a genuinely distinctive fact (Little Rock is the
  state capital) — this is a case where the model over-applied the boilerplate rule the prompt
  warned against.
- No fabricated facts observed in any of the three cleaned outputs — all retained sentences
  are verbatim or near-verbatim from the source text, consistent with the prompt's
  no-inference constraint.
