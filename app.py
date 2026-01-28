import streamlit as st
import psycopg2
import random
import json
import asyncio
from datetime import date
from typing import Dict, Any, List

from openai import AsyncAzureOpenAI, APIError, RateLimitError
from httpx import ReadTimeout


INTENTS = [
    "story",
    "rant",
    "ask advice",
    "get clarity",
    "update",
    "alert",
]

CATEGORIES = [
    "Sex Education",
    "Infertility",
    "Parenting",
    "Men’s Health",
    "PCOD",
    "Hair Care",
    "Heart Health",
    "Sexual Health",
    "Weight Loss",
]


REQUIRED_SECRETS = [
    "HEALTHCARE_DB_HOST",
    "HEALTHCARE_DB_PORT",
    "HEALTHCARE_DB_USER",
    "HEALTHCARE_DB_PASS",
    "HEALTHCARE_DB_BASE",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
]

for key in REQUIRED_SECRETS:
    if key not in st.secrets:
        st.error(f"Missing required secret: {key}")
        st.stop()


def get_db():
    return psycopg2.connect(
        host=st.secrets["HEALTHCARE_DB_HOST"],
        port=st.secrets["HEALTHCARE_DB_PORT"],
        user=st.secrets["HEALTHCARE_DB_USER"],
        password=st.secrets["HEALTHCARE_DB_PASS"],
        dbname=st.secrets["HEALTHCARE_DB_BASE"],
        sslmode="require",
    )


def compute_age_group(dob):
    if not dob:
        return "25-35"

    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    if age < 25:
        return "20-30"
    elif age < 35:
        return "25-35"
    else:
        return "35+"


def fetch_bot_users():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    username,
                    date_of_birth,
                    gender,
                    tone,
                    typing_quirks
                FROM users
                WHERE
                    account_type = 'user'
                    AND tone IS NOT NULL
                ORDER BY username
            """
            )
            rows = cur.fetchall()

        users = []

        for r in rows:
            users.append(
                {
                    "id": r[0],
                    "username": r[1],
                    "age_group": compute_age_group(r[2]),
                    "gender": r[3],
                    "tone": r[4],
                    "typing_quirks": r[5],
                }
            )
        return users
    finally:
        conn.close()


def fetch_angles(intent: str) -> List[str]:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ia.angle
                FROM intent_angles ia
                JOIN intents i ON i.id = ia.intent_id
                WHERE i.name = %s
            """,
                (intent,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def fetch_simple_names(table: str) -> List[str]:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT name FROM {table} ORDER BY name")
            rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def fetch_vibe_profiles() -> Dict[str, Dict[str, Any]]:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, description, emoji_chance
                FROM vibe_profiles
            """
            )
            rows = cur.fetchall()

        return {
            r[0]: {
                "description": r[1],
                "emoji_chance": r[2],
            }
            for r in rows
        }
    finally:
        conn.close()


MENTAL_STATES = fetch_simple_names("mental_states")
COGNITIVE_BIASES = fetch_simple_names("cognitive_biases")

VIBE_PROFILES = {
    "genz": {"emoji_chance": 0.25},
    "millennial": {"emoji_chance": 0.05},
    "genx": {"emoji_chance": 0.0},
}


client = AsyncAzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    timeout=30.0,
    max_retries=0,
)

MODEL = "gpt-4o"


async def _call_with_retry(fn, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            return await fn()
        except (RateLimitError, APIError, ReadTimeout):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(2**attempt)


def assign_vibe(persona: Dict[str, Any]) -> Dict[str, Any]:
    persona = persona.copy()
    age_group = persona.get("age_group")

    if age_group == "20-30":
        vibe = "genz"
    elif age_group == "25-35":
        vibe = "millennial"
    else:
        vibe = "genx"

    persona["vibe"] = vibe
    persona["vibe_profile"] = VIBE_PROFILES.get(vibe, {})
    return persona


def degrade_persona(persona: Dict[str, Any], intent: str) -> Dict[str, Any]:
    persona = persona.copy()
    persona["intent"] = intent
    persona["mental_state"] = (
        random.choice(MENTAL_STATES) if MENTAL_STATES else "neutral"
    )

    persona["max_tokens"] = 140
    return persona


async def generate_post_blueprint(input_data: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
You are simulating a REAL user of a healthcare social app.

Return JSON with EXACT keys:
{
  "title": "short, hesitant, imperfect title",
  "user_belief": "one sentence belief the user is stuck on",
  "emotional_state": "one word",
  "keywords": ["string"]
}

Rules:
- Title must feel typed, not written
- Avoid medical jargon
- Title and belief MUST refer to the same issue
- Belief reflects confusion or fear
"""

    async def _call():
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(input_data)},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        return json.loads(r.choices[0].message.content)

    return await _call_with_retry(_call)


async def apply_cognitive_distortion(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    if not COGNITIVE_BIASES:
        bias = "general_uncertainty"
    else:
        bias = random.choice(COGNITIVE_BIASES)

    system_prompt = f"""
Rewrite the belief with a subtle cognitive distortion: {bias}

Return JSON:
{{ "distorted_belief": "string" }}

Rules:
- Meaning must stay related
- Slightly increase fear or confusion
"""

    async def _call():
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": blueprint["user_belief"]},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        return json.loads(r.choices[0].message.content)

    raw = await _call_with_retry(_call)
    blueprint["distorted_belief"] = raw.get(
        "distorted_belief",
        blueprint["user_belief"],
    )
    blueprint["bias"] = bias
    return blueprint


import requests


def search_gifs(query: str, limit: int = 10):
    GIPHY_BASE_URL = "https://api.giphy.com/v1/gifs/search"
    GIPHY_API_KEY = st.secrets["GIPHY_API_KEY"]

    params = {
        "api_key": GIPHY_API_KEY,
        "q": query,
        "limit": limit,
        "rating": "g",
        "lang": "en",
    }

    response = requests.get(GIPHY_BASE_URL, params=params, timeout=5)
    response.raise_for_status()

    data = response.json()
    gifs = []

    for item in data.get("data", []):
        gifs.append(
            {
                "id": item["id"],
                "url": item["images"]["original"]["url"],
                "preview": item["images"]["fixed_height_small"]["url"],
            }
        )

    return gifs


async def realize_language(
    blueprint: Dict[str, Any],
    persona: Dict[str, Any],
) -> str:
    system_prompt = """
You are writing a REAL post for a healthcare app.

Rules:
- DO NOT repeat the title
- First person only
- No advice, no conclusions
- Imperfect grammar allowed
- Sound like typing, not writing
"""

    user_payload = {
        "belief": blueprint["distorted_belief"],
        "intent": persona["intent"],
        "mental_state": persona["mental_state"],
        "typing_quirks": persona.get("typing_quirks"),
        "vibe": persona.get("vibe"),
    }

    async def _call():
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.85,
            max_tokens=persona["max_tokens"],
        )
        return r.choices[0].message.content.strip()

    return await _call_with_retry(_call)


def should_attach_poll(idx: int) -> bool:
    return idx % 4 == 3


async def generate_value_poll(
    blueprint: Dict[str, Any],
    category: str,
    intent: str,
) -> Dict[str, Any]:
    system_prompt = """
You are creating a poll attached to a user's post.

Return JSON:
{
  "question": "internal dilemma question",
  "options": ["option1", "option2", "option3"]
}
"""

    async def _call():
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "belief": blueprint["distorted_belief"],
                            "intent": intent,
                            "category": category,
                        }
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        return json.loads(r.choices[0].message.content)

    raw = await _call_with_retry(_call)
    return {
        "question": raw.get("question", "am i overthinking this?"),
        "options": raw.get("options", [])[:3],
    }


async def generate_ai_post(user, intent, category, attach_gif):
    angles = fetch_angles(intent)
    angle = random.choice(angles) if angles else "general"

    blueprint = await generate_post_blueprint(
        {
            "category": category,
            "intent": intent,
            "angle": angle,
            "bot_persona": user,
        }
    )

    blueprint = await apply_cognitive_distortion(blueprint)

    persona = assign_vibe(user)
    persona = degrade_persona(persona, intent)

    description = await realize_language(blueprint, persona)

    poll = None
    if should_attach_poll(random.randint(0, 9)):
        poll = await generate_value_poll(blueprint, category, intent)

    # media = None
    # gif = select_gif_for_post(blueprint)
    # if gif:
    #     media = [gif]

    tags = (
        [intent, category.lower().replace(" ", "")] + blueprint.get("keywords", [])
    )[:5]

    medias = []

    # ---- GIF ATTACHMENT ----
    if attach_gif:
        keyword = tags
        gifs_list = search_gifs(keyword, limit=6)

        if gifs_list:
            gif_selected = random.choice(gifs_list)
            medias.append(
                {
                    "type": "gif",
                    "uri": gif_selected["url"],
                    "preview": gif_selected["preview"],
                    "width": 220,
                    "height": 235,
                    "aspectRatio": 0.9361702127659575,
                    "orientation": "portrait",
                }
            )

    return {
        "user_id": user["id"],
        "intent": intent,
        "category": category,
        "title": blueprint["title"],
        "description": description,
        "poll": poll,
        "tags": tags,
        "medias": medias or None,
    }


def run_ai(user, intent, category, attach_gif):
    return asyncio.run(generate_ai_post(user, intent, category, attach_gif))


def save_post_to_db(post: Dict[str, Any], show_public: bool):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO posts (
                    user_id,
                    category,
                    intent,
                    title,
                    description,
                    tags,
                    show_public,
                    medias
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    post["user_id"],
                    post["category"],
                    post["intent"],
                    post["title"],
                    post["description"],
                    post["tags"],
                    show_public,
                    post["medias"],
                ),
            )
            conn.commit()
    finally:
        conn.close()


def upsert_user(
    username: str,
    dob,
    gender: str,
    tone: str,
    typing_quirks: str,
):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (
                    username,
                    date_of_birth,
                    gender,
                    tone,
                    typing_quirks,
                    account_type
                )
                VALUES (%s, %s, %s, %s, %s, 'user')
                ON CONFLICT (username)
                DO UPDATE SET
                    date_of_birth = EXCLUDED.date_of_birth,
                    gender = EXCLUDED.gender,
                    tone = EXCLUDED.tone,
                    typing_quirks = EXCLUDED.typing_quirks
            """,
                (
                    username,
                    dob,
                    gender,
                    tone,
                    typing_quirks,
                ),
            )
            conn.commit()
    finally:
        conn.close()


st.set_page_config(
    page_title="Healthcare AI Post Generator",
    layout="wide",
)
st.title(" Healthcare AI Post Generator ")


users = fetch_bot_users()

if not users:
    st.warning("No bot users with tone found in DB.")
    st.stop()


tabs = st.tabs(
    [
        "👤 Users",
        "🎯 Angles",
        "🧠 Biases",
        "💭 Mental States",
        "🎭 Vibe Profiles",
        "🧪 Generate Post",
    ]
)


with tabs[0]:
    st.subheader("👤 Add / Update Bot Users")

    conn = get_db()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, username, date_of_birth, gender, tone, typing_quirks
            FROM users
            WHERE tone IS NOT NULL
            ORDER BY username
            """
        )
        rows = cur.fetchall()
    conn.close()

    user_map = {r[1]: r for r in rows}
    usernames = list(user_map.keys())

    selected = st.selectbox(
        "Select existing user (optional)",
        [""] + usernames,
    )

    if selected:
        uid, uname, dob, gender, tone, quirks = user_map[selected]
    else:
        uid = None
        uname = ""
        dob = None
        gender = ""
        tone = ""
        quirks = ""

    # ---- INPUTS (SOURCE OF TRUTH) ----
    username = st.text_input("Username", value=uname)
    dob = st.date_input("Date of birth", value=dob)
    gender = st.selectbox(
        "Gender",
        ["", "female", "male", "other"],
        index=["", "female", "male", "other"].index(gender or ""),
    )
    tone = st.text_area("Tone", value=tone)
    quirks = st.text_area("Typing quirks", value=quirks)

    # ---- SAVE ----
    if st.button("Save user"):
        if not username or not tone or not quirks:
            st.error("Username, tone, and typing quirks are required")
        else:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, date_of_birth, gender, tone, typing_quirks)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (username)
                    DO UPDATE SET
                        date_of_birth = EXCLUDED.date_of_birth,
                        gender = EXCLUDED.gender,
                        tone = EXCLUDED.tone,
                        typing_quirks = EXCLUDED.typing_quirks
                    """,
                    (username, dob, gender, tone, quirks),
                )
                conn.commit()
            conn.close()

            st.success("User saved")
            st.experimental_rerun()


with tabs[1]:
    st.subheader("Intent Angles")

    intent = st.selectbox("Select intent", INTENTS)
    angles = fetch_angles(intent)

    st.write("Existing angles:")
    for a in angles:
        st.code(a)

    new_angle = st.text_input("Add new angle")
    if st.button("Add angle"):
        if new_angle:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO intent_angles (intent_id, angle)
                    SELECT id, %s FROM intents WHERE name = %s
                """,
                    (new_angle, intent),
                )
                conn.commit()
            conn.close()
            st.success("Angle added")
            st.experimental_rerun()


with tabs[2]:
    st.subheader("Cognitive Biases")

    # --- FETCH FROM DB ---
    biases = fetch_simple_names("cognitive_biases")
    print(biases)

    # --- DISPLAY EXISTING ---
    st.markdown("### Existing cognitive biases (from DB)")
    if not biases:
        st.warning("No cognitive biases found. Add at least one.")
    else:
        for b in biases:
            st.write(f"• {b}")

    st.divider()

    # --- ADD NEW ---
    st.markdown("### Add new cognitive bias")
    new_bias = st.text_input("Cognitive bias name")

    if st.button("Add cognitive bias"):
        if not new_bias:
            st.error("Bias name cannot be empty")
        else:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO cognitive_biases (name) VALUES (%s)",
                    (new_bias,),
                )
                conn.commit()
            conn.close()
            st.success("Cognitive bias added")
            st.experimental_rerun()

    st.divider()

    # --- DELETE ---
    st.markdown("### Delete cognitive bias")
    bias_to_delete = st.selectbox(
        "Select bias to delete",
        [""] + biases,
    )

    if st.button("Delete cognitive bias"):
        if not bias_to_delete:
            st.error("Select a bias")
        else:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM cognitive_biases WHERE name=%s",
                    (bias_to_delete,),
                )
                conn.commit()
            conn.close()
            st.success("Bias deleted")
            st.experimental_rerun()


with tabs[3]:
    st.subheader("Mental States")

    # --- FETCH FROM DB ---
    states = fetch_simple_names("mental_states")

    # --- DISPLAY EXISTING ---
    st.markdown("### Existing mental states (from DB)")
    if not states:
        st.warning("No mental states found. Add at least one.")
    else:
        for s in states:
            st.write(f"• {s}")

    st.divider()

    # --- ADD NEW ---
    st.markdown("### Add new mental state")
    new_state = st.text_input("Mental state name")

    if st.button("Add mental state"):
        if not new_state:
            st.error("Mental state name cannot be empty")
        else:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO mental_states (name) VALUES (%s)",
                    (new_state,),
                )
                conn.commit()
            conn.close()
            st.success("Mental state added")
            st.experimental_rerun()

    st.divider()

    # --- DELETE ---
    st.markdown("### Delete mental state")
    state_to_delete = st.selectbox(
        "Select mental state to delete",
        [""] + states,
    )

    if st.button("Delete mental state"):
        if not state_to_delete:
            st.error("Select a mental state")
        else:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM mental_states WHERE name=%s",
                    (state_to_delete,),
                )
                conn.commit()
            conn.close()
            st.success("Mental state deleted")
            st.experimental_rerun()

with tabs[4]:
    st.subheader("Vibe Profiles")

    vibes = fetch_vibe_profiles()

    st.markdown("### Existing vibe profiles (from DB)")
    if not vibes:
        st.warning("No vibe profiles found.")
    else:
        for name, v in vibes.items():
            with st.expander(name):
                st.write("Description:", v["description"])
                st.write("Emoji chance:", v["emoji_chance"])

    st.divider()

    st.markdown("### Add / Update vibe profile")

    existing_vibes = ["genz", "millennial", "genx"]

    selected_vibe = st.selectbox(
        "Select existing vibe (optional)",
        [""] + existing_vibes,
    )
    vibe_name = st.text_input(
        "Vibe name",
        value=selected_vibe,
    )

    description = st.text_area("Description")
    emoji_chance = st.slider(
        "Emoji chance",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
    )

    if st.button("Save vibe profile"):
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vibe_profiles (name, description, emoji_chance)
                VALUES (%s, %s, %s)
                ON CONFLICT (name)
                DO UPDATE SET
                    description = EXCLUDED.description,
                    emoji_chance = EXCLUDED.emoji_chance
            """,
                (vibe_name, description, emoji_chance),
            )
            conn.commit()
        conn.close()
        st.success("Vibe profile saved")
        st.experimental_rerun()


with tabs[5]:
    st.subheader("Generate AI Post")

    user = st.selectbox(
        "Select bot user",
        users,
        format_func=lambda u: u["username"],
    )

    intent = st.selectbox(
        "Intent (optional)",
        ["Random"] + INTENTS,
    )

    category = st.selectbox(
        "Category (optional)",
        ["Random"] + CATEGORIES,
    )

    if intent == "Random":
        intent = random.choice(INTENTS)

    if category == "Random":
        category = random.choice(CATEGORIES)

    attach_gif = st.checkbox("Attach GIF", value=False)

    if st.button("✨ Generate Post"):
        with st.spinner("Generating..."):
            post = run_ai(user, intent, category, attach_gif)
            st.session_state["generated_post"] = post


if "generated_post" in st.session_state:
    post = st.session_state["generated_post"]

    st.markdown("### 📝 Preview")
    st.markdown(f"**Title:** {post['title']}")
    st.write(post["description"])
    if post.get("medias"):
        for media in post["medias"]:
            if media["type"] == "gif":
                st.image(
                    media["uri"],
                    width=media.get("width", 240),
                )
    st.caption(f"Intent: {post['intent']} | Category: {post['category']}")

    show_public = st.checkbox("Show public", value=False)

    if st.button("💾 Save to DB"):
        save_post_to_db(post, show_public)
        st.success("Post saved")
        del st.session_state["generated_post"]
