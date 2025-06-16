import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from textwrap import dedent

st.title("One Piece YouTube Shorts AI Agent 🏴‍☠️")
st.caption("Generate viral One Piece YouTube shorts scripts about unfamiliar abilities and hidden stories")

openai_api_key = st.text_input("Enter OpenAI API Key to access GPT-4o", type="password")
serper_api_key = st.text_input("Enter Serper API Key for Search functionality", type="password")

if openai_api_key and serper_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    search_tool = SerperDevTool(api_key=serper_api_key)
    scrape_tool = ScrapeWebsiteTool()

    # Topic Searcher Agent
    topic_searcher = Agent(
        role="One Piece Topic Researcher",
        goal="Find the most intriguing, lesser-known facts about One Piece characters and abilities for viral content",
        backstory=dedent("""
            You are a One Piece expert and viral content researcher with deep knowledge of the series.
            Your specialty is uncovering hidden gems - characters with surprising abilities, untold backstories,
            and mind-blowing connections that most fans don't know about. You know exactly what makes
            viewers stop scrolling and watch a YouTube short.
        """),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # Script Writer Agent
    script_writer = Agent(
        role="Viral YouTube Shorts Script Writer",
        goal="Transform One Piece wiki content into highly engaging 60-second YouTube shorts scripts",
        backstory=dedent("""
            You are a master YouTube Shorts scriptwriter who specializes in One Piece content.
            You have a proven track record of creating viral videos with millions of views.
            Your scripts use dramatic contrast, humor, and specific lore facts to keep viewers
            hooked from the first second to the last. You know the exact formula that makes
            One Piece content go viral.
        """),
        tools=[scrape_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # Editor Agent
    editor = Agent(
        role="Content Editor and Quality Assurance",
        goal="Polish and optimize One Piece YouTube shorts scripts for maximum engagement and accuracy",
        backstory=dedent("""
            You are a meticulous content editor with expertise in viral YouTube content.
            You have an eye for detail and know exactly what separates good content from
            viral content. Your job is to ensure every script is grammatically perfect,
            factually accurate, and optimized for maximum viewer retention and engagement.
        """),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )

    # Define Tasks
    def create_tasks(user_request=""):
        # Task 1: Topic Research
        if user_request:
            search_query = f"Find ONE specific One Piece Wiki character page about {user_request} with surprising abilities or hidden stories"
        else:
            search_query = "Find ONE compelling One Piece Wiki character page with lesser-known facts, unfamiliar abilities, or hidden stories"
        
        topic_research_task = Task(
            description=dedent(f"""
                Search for ONE specific One Piece character or topic that would make viewers stop scrolling.
                
                Instructions:
                - Search for: {search_query}
                - Use targeted searches like: 'site:onepiece.fandom.com [specific character name]' or 'site:onepiece.fandom.com [specific devil fruit]'
                - Focus on finding ONE compelling character page with surprising information
                - Look for characters with unusual abilities, hidden backstories, or surprising connections
                - Prioritize characters with surprising size comparisons, hidden powers, or unexpected backstories
                - Examples: Sanjuan Wolf, Gedatsu, Shiki, lesser-known giants, characters with unusual devil fruits
                
                Return:
                - ONE specific One Piece Wiki URL
                - Brief explanation of why this character/topic is compelling for a viral short
                - Key surprising facts that make this worth covering
            """),
            agent=topic_searcher,
            expected_output="A single One Piece Wiki URL with explanation of why it's viral-worthy and key surprising facts"
        )

        # Task 2: Script Writing
        script_writing_task = Task(
            description=dedent("""
                Using the One Piece Wiki URL provided, create a viral YouTube Shorts script.
                
                Process:
                1. Read the wiki page using the scrape website tool
                2. Extract the most surprising and engaging facts
                3. Write a script following the EXACT format below
                
                FORMAT TO FOLLOW EXACTLY:
                
                **SHORT [#]: [CATCHY TITLE]**
                
                [Start with a common belief]
                
                But [twist or contradiction].
                
                **[Character Name]** [main trait or reveal].
                
                [3–5 short facts, each on its own line]
                
                [Final twist, cliffhanger, or surprise]
                
                STRICT RULES:
                - Line breaks after EVERY sentence
                - Each sentence: under 10 words
                - Max 120 words total
                - Bold character names and important terms
                - No labels like 'HOOK:', 'FACTS:', etc.
                - Must sound like a YouTube narrator
                - Be dramatic, funny, or surprising
                - End on a punchline, mystery, or big twist
                
                Study these examples for style:
                - "Wadatsumi is one of the biggest character... But this guy towers over him by 300 feet."
                - "**Boa Hancock** is the most beautiful woman... But 38 years ago, **Gloriosa** held that same title."
                - "**Luffy** might be the dumbest captain... But **Luffy** is basically Einstein compared to this guy."
            """),
            agent=script_writer,
            expected_output="A viral YouTube Shorts script following the exact format with dramatic hooks and surprising reveals",
            context=[topic_research_task]
        )

        # Task 3: Editing and Polishing
        editing_task = Task(
            description=dedent("""
                Edit and polish the YouTube Shorts script for maximum engagement.
                
                Check for:
                
                Grammar & Flow:
                - Perfect grammar and punctuation
                - Smooth transitions between sentences
                - Consistent tense throughout
                - Proper capitalization of character names and abilities
                
                Engagement Optimization:
                - Compelling hook that creates curiosity
                - Facts are accurate to the wiki page
                - Script builds suspense effectively
                - Ending provides satisfying revelation
                
                Technical Requirements:
                - Script is 80-120 words maximum
                - Very short, punchy sentences
                - Bold formatting for character names only
                - Line breaks after each sentence
                - NO section headers (HOOK, CONTRADICTION, etc.)
                
                Content Quality:
                - Information is accurate and verifiable
                - Topic is genuinely surprising or lesser-known
                - Script follows the proven viral format
                - Ending creates desire for more content
                
                Return the final polished script ready for recording.
            """),
            agent=editor,
            expected_output="A polished, grammatically perfect YouTube Shorts script optimized for maximum engagement",
            context=[topic_research_task, script_writing_task]
        )

        return [topic_research_task, script_writing_task, editing_task]

    user_request = st.text_input("Any specific One Piece character or topic you want to focus on? (Leave blank for AI to choose)")

    if st.button("Generate One Piece Short Script"):
        with st.spinner("🔍 Searching for viral One Piece topics..."):
            try:
                tasks = create_tasks(user_request)
                crew = Crew(
                    agents=[topic_searcher, script_writer, editor],
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True
                )
                result = crew.kickoff()
                st.markdown("## Your One Piece YouTube Short Script:")
                st.markdown("---")
                st.write(result)
                st.markdown("---")
                st.caption("💡 Tip: This script is optimized for 45-60 seconds of reading time. Practice your delivery for maximum engagement!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API keys and try again.")

else:
    st.info("Please enter both API keys to get started.")
    st.markdown("""
    ### Required API Keys:
    1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
    2. **Serper API Key**: Get from [Serper.dev](https://serper.dev/) (Free tier available)
    """)

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

st.sidebar.markdown("## 🛠️ Installation Requirements")
st.sidebar.markdown("""
```bash
pip install crewai
pip install crewai-tools
pip install langchain-openai
pip install streamlit
```
""")
