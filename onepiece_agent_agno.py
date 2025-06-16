# Import the required libraries
from textwrap import dedent
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.tools.website import WebsiteTools
import streamlit as st
from agno.models.openai import OpenAIChat


st.title("One Piece YouTube Shorts AI Agent 🏴‍☠️")
st.caption("Generate viral One Piece YouTube shorts scripts about unfamiliar abilities and hidden stories")

openai_api_key = st.text_input("Enter OpenAI API Key to access GPT-4o", type="password")
serp_api_key = st.text_input("Enter Serp API Key for Search functionality", type="password")

if openai_api_key and serp_api_key:
    # Topic Searcher Agent
    topic_searcher = Agent(
        name="OnePieceTopicSearcher",
        role="Searches for hookable One Piece topics about unfamiliar abilities and hidden stories",
        model=OpenAIChat(id="gpt-4.1-mini", api_key=openai_api_key),
        description=dedent(
            """\
        You are a One Piece expert and viral content researcher. Your job is to find the most 
        intriguing, lesser-known facts about One Piece characters, their abilities, and hidden stories 
        that would make viewers stop scrolling and watch a YouTube short.
        """
        ),
        instructions=[
            "Search for ONE specific One Piece character or topic at a time to avoid token limits.",
            "Use targeted searches like: 'site:onepiece.fandom.com [specific character name]' or 'site:onepiece.fandom.com [specific devil fruit]'",
            "Focus on finding ONE compelling character page that has surprising or lesser-known information.",
            "Look for characters with unusual abilities, hidden backstories, or surprising connections.",
            "From search results, select the SINGLE most promising One Piece Wiki character page URL.",
            "Avoid general searches - be specific about one character or ability.",
            "Return only ONE wiki URL with a brief explanation of why it's compelling for a viral short.",
            "Prioritize characters that have surprising size comparisons, hidden powers, or unexpected backstories.",
            "Examples of good targets: Sanjuan Wolf, Gedatsu, Shiki, lesser-known giants, characters with unusual devil fruits."
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )

    # Script Writer Agent
    script_writer = Agent(
        name="OnePieceScriptWriter",
        role="Writes highly engaging One Piece YouTube Shorts scripts using dramatic contrast, humor, and specific lore facts",
        model=OpenAIChat(id="gpt-4.1-mini", api_key=openai_api_key),
        description=dedent(
            """\
            You are a viral YouTube Shorts scriptwriter specializing in One Piece content.
            You write dramatic, funny, and hook-driven scripts using short punchy lines and real canon details.
            Your job is to turn wiki pages into 60-second script gold.
            """
        ),
        instructions=[
            "You will receive ONE specific One Piece Wiki URL from the topic searcher.",
            "Use `read_website()` to pull **only essential details** — especially abilities, sizes, fruits, ranks, and backstories.",
            "",
            "Follow this structure *exactly*:",
            "",
            "**SHORT [#]: [CATCHY TITLE]**",
            "",
            "[Start with a common belief]",
            "",
            "But [twist or contradiction].",
            "",
            "**[Character Name]** [main trait or reveal].",
            "",
            "[3–5 short facts, each on its own line]",
            "",
            "[Final twist, cliffhanger, or surprise that makes the viewer want more]",
            "",
            "⚠️ RULES TO FOLLOW:",
            "- Use line breaks after EVERY sentence.",
            "- Each sentence: under 10 words.",
            "- Max 120 words total.",
            "- Bold character names and important terms.",
            "- No labels like 'HOOK:', 'FACTS:', etc.",
            "- Must sound like a YouTube narrator—not a wiki.",
            "- Be dramatic, funny, or surprising.",
            "- End on a punchline, mystery, or big twist.",
            "",
            "✅ EXACT STYLE EXAMPLES TO FOLLOW:",
            "",
            "**SHORT 27: The BIGGEST Character in One Piece**",
            "",
            "Wadatsumi is one of the biggest character in One Piece...",
            "",
            "But this guy towers over him by 300 feet.",
            "",
            "**Sanjuan Wolf** is not your ordinary giant.",
            "",
            "Because aside from being a giant, he ate the Deka Deka no Mi...",
            "",
            "...making him the tallest character in One Piece.",
            "",
            "He's even bigger than the Statue of Liberty, and nearly half the height of the Eiffel Tower.",
            "",
            "After defeating Bonney on the burning island,he stood on the ocean floor to avoid getting on fire.",
            "",
            "Although the sea weakens him,he didn’t drown—because of his sheer size.",
            "",
            "Interestingly, the Mini Mini no Mi,which is the counterpart to the Deka Deka no Mi, was eaten by another giant—Lily—",
            "",
            "making her the smallest giant in One Piece.",
            "",
            "---",
            "",
            "**SHORT 23: The most beautiful woman in One Piece**",
            "",
            "**Boa Hancock** is the most beautiful woman in One Piece.",
            "",
            "But 38 years ago, **Gloriosa** held that same title.",
            "",
            "She’s now old and short,yet she was once the Pirate Empress—three generations before Hancock.",
            "",
            "She left her kingdom because of lovesickness.",
            "",
            "And followed the strongest man she met across the seas—This man happened to be a member of the legendary **Rocks Pirates**.",
            "",
            "Well, that man… might be **Kaido**.",
            "",
            "**Yamato’s** potential mother has long been one of the biggest mysteries in One Piece.",
            "",
            "But **Oda** actually foreshadowed this connection way back on Amazon Lily,when **Luffy** showed **Ace’s vivre card** to the Kuja Pirates.",
            "",
            "Coincidentally, that vivre card was created by **Yamato** and given to **Ace**.",
            "",
            "**Gloriosa** didn’t just randomly know about vivre cards—She knew because her own son **Yamato** had mastered the technique.",
            "",
            "And now, we also know what **Nami** will look like after 50 years.",
            "",
            "---",
            "",
            "**SHORT 22: The Dumbest Character in One Piece**",
            "",
            "**Luffy** might be the dumbest captain in One Piece...",
            "",
            "But if we're talking about the dumbest character overall—",
            "",
            "**Luffy** is basically Einstein compared to this guy.",
            "",
            "And that guy is **Gedatsu**.",
            "",
            "**Gedatsu** is so stupid that he literally forgets to breathe.",
            "",
            "This dude rolls his eyes so far back into his head that he can't see anything...",
            "",
            "And then he wonders why the world went dark, or why he thinks **Chopper** turned invisible.",
            "",
            "But wait—it gets worse.",
            "",
            "He once tried to enter a house through the window... when the door was wide open right next to him.",
            "",
            "He puts food in his ears instead of his mouth during meals.",
            "",
            "And here’s my favorite:",
            "",
            "While **Luffy** once endangered his crew by nearly crashing into a waterfall...",
            "",
            "**Gedatsu** accidentally aimed his attack at an ally instead of his enemy.",
            "",
            "Yes, **Luffy** is an idiot—but at least he remembers basic human functions like breathing and blinking.",
            "",
            "**Gedatsu** literally has to remind himself to do both.",
        ],
        tools=[WebsiteTools()],
        add_datetime_to_instructions=True,
        markdown=True,
    )

    # Editor Agent
    editor = Agent(
        name="ScriptEditor",
        role="Edits and polishes One Piece YouTube shorts scripts for maximum engagement",
        model=OpenAIChat(id="gpt-4.1-mini", api_key=openai_api_key),
        team=[topic_searcher, script_writer],
        description=dedent(
            """\
        You are a YouTube shorts content editor specializing in One Piece viral content. 
        Your job is to ensure scripts are grammatically perfect, engaging, and optimized for retention.
        """
        ),
        instructions=[
            "Ask the topic searcher to find ONE specific One Piece Wiki page about a character with surprising abilities or hidden stories.",
            "Then, provide the single wiki URL to the script writer to create a viral short script.",
            "Finally, edit the script for:",
            "",
            "Grammar & Flow:",
            "- Perfect grammar and punctuation",
            "- Smooth transitions between sentences",
            "- Consistent tense throughout",
            "- Proper capitalization of character names and abilities",
            "",
            "Engagement Optimization:",
            "- Ensure the hook is compelling and creates curiosity",
            "- Verify all facts are accurate to the wiki page",
            "- Check that the script builds suspense effectively",
            "- Confirm the ending provides a satisfying revelation",
            "",
            "Technical Requirements:",
            "- Script must be 80-120 words maximum",
            "- Very short, punchy sentences",
            "- Bold formatting for character names only",
            "- Line breaks after each sentence",
            "- NO section headers (HOOK, CONTRADICTION, etc.)",
            "",
            "Content Quality:",
            "- Information is accurate and verifiable from the specific wiki page",
            "- Topic is genuinely surprising or lesser-known",
            "- Script follows the proven viral format from examples",
            "- Ending creates desire to watch more content",
            "- All facts are sourced from the official wiki content",
            "",
            "Token Management:",
            "- Keep responses concise and focused",
            "- Don't repeat large blocks of text",
            "- Focus on the most impactful edits only"
        ],
        add_datetime_to_instructions=True,
        markdown=True,
    )

    user_request = st.text_input("Any specific One Piece character or topic you want to focus on? (Leave blank for AI to choose)")

    if st.button("Generate One Piece Short Script"):
        with st.spinner("🔍 Searching for viral One Piece topics..."):
            if user_request:
                query = f"Find ONE specific One Piece Wiki character page about {user_request} that has surprising abilities or hidden stories perfect for a viral YouTube short"
            else:
                query = "Find ONE compelling One Piece Wiki character page with lesser-known facts, unfamiliar abilities, or hidden stories that would make a viral YouTube short"
            response = editor.run(query, stream=False)
            
            st.markdown("## Your One Piece YouTube Short Script:")
            st.markdown("---")
            st.write(response.content)
            st.markdown("---")
            st.caption("💡 Tip: This script is optimized for 45-60 seconds of reading time. Practice your delivery for maximum engagement!")


# Add sidebar with tips
st.sidebar.markdown("## 🎯 Script Success Tips")
st.sidebar.markdown("""
**For Best Results:**
- Speak clearly and with enthusiasm
- Pause at ellipses (...) for dramatic effect
- Emphasize bolded words and names
- Use hand gestures for size comparisons
- End with energy to encourage engagement

**Viral Potential Indicators:**
- ✅ Starts with surprising contradiction
- ✅ Includes specific numbers/measurements  
- ✅ Reveals lesser-known canon information
- ✅ Ends with cliffhanger or revelation
- ✅ Under 60 seconds reading time
""")

st.sidebar.markdown("## 📝 Script Format Example")
st.sidebar.markdown("""
**Correct Format:**
```
**SHORT 28: Title Here**

Luffy might be the strongest...

But this guy is actually stronger.

**Character Name** ate the X-X fruit.

He's 300 feet tall.

He destroyed half a city.

And walked away with just bruises.

This guy's story is far from over.
```

**❌ Wrong Format:**
- Long paragraphs
- Section headers (HOOK:, etc.)
- Verbose explanations
- Over 120 words
""")

st.sidebar.markdown("## 📊 Example Performance")
st.sidebar.markdown("""
**Top performing elements:**
- Size comparisons (Eiffel Tower, Statue of Liberty)
- "But this guy..." reveals
- Specific numbers (300 feet, 51 divisions)
- Hidden connections between characters
- Abilities that defy expectations
""")
