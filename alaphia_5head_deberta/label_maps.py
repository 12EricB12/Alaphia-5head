#!/usr/bin/env python3
"""Label taxonomies and id mappings for the 5-head Alaphia multitask model (DeBERTa).

Single source of truth for training and inference. Built from the same emotion/need
taxonomies as GPT_Classification batch scripts; life contexts align with label_maps.json order.
"""
from __future__ import annotations

from typing import Dict, List

# --- Raw label lists (class index = position in list) ---
EMOTION_LABELS: List[str] = ["Dread", "Anxiety", "Nervous", "Rage", "Anger", "Irritation", "Humiliated Rage", "Frustration", "Resentment", "Ruefulness", "Fed Up", "Bitterness", "Mortification", "Shame", "Self-consciousness", "Cringe", "Guilt", "Regret", "Remorse", "Angry regret", "Despair", "Sadness", "Dismay", "Hurt", "Desperation", "Burnout", "Exhaustion", "Tiredness", "Overwhelm", "Panic", "Fear", "Unease", "Vulnerable", "Insecure", "Helpless", "Shaky", "Envy", "Jealousy", "Revulsion", "Disgust", "Aversion", "Restlessness", "Boredom", "Indifference", "Elation", "Happiness", "Gladness", "Amusement", "Liberation", "Relief", "Awe", "Admiration", "Gratitude", "Triumph", "Pride", "Confidence", "Serenity", "Calm", "Mellow", "Shock", "Astonishment", "Surprise", "Nostalgia", "Longing", "Bittersweet", "Excitement", "Optimism", "Eagerness", "Anticipation", "Interest", "Ambivalence", "Torn"]

NEED_LABELS: List[str] = ["Competence", "Accomplishment", "Effectiveness", "Growth", "Mastery", "Strength", "Impact", "Hope", "Progress", "Challenge", "Independence", "Authenticity", "Choice", "Space", "Boundaries", "Freedom", "Participation", "Self-expression", "Autonomy", "Knowledge", "Creativity", "Adventure", "Discovery", "Stimulation", "Acceptance", "Inclusion", "Support", "Companionship", "Connection", "Affection", "Belonging", "Community", "Visibility", "Love", "Attention", "Intimacy", "Nurturance", "Care", "Trust", "Order", "Security", "Predictability", "Physical safety", "Emotional safety", "Shelter", "Stability", "Protection", "Reassurance", "Ease", "Transparency", "Justice", "Accountability", "Social Justice", "Equity", "Relational Balance", "Fairness", "Equality", "Vindication", "Redemption", "Replenishment", "Healing", "Forgiveness", "Wholeness", "Closure", "Self-Respect", "Loyalty", "Self-Honesty", "Dignity", "Integrity", "Dependability", "Appreciation", "Recognition", "Respect", "Clarity", "Esteem", "Empathy", "Compassion", "Acknowledgement", "Validation", "Understanding", "Tranquility", "Harmony", "Solitude", "Reflection", "Beauty", "Peace", "Exertion", "Rest", "Physical", "Pleasure", "Nourishment", "Touch", "Vitality", "Energy", "Sexual expression", "Humour", "Fun", "Play", "Laughter", "Spontaneity", "Leisure", "Engagement", "Mourning", "Awareness", "Inspiration", "Celebration", "Contribution", "Meaning", "Legacy", "Transcendence", "Generativity", "Purpose"]

MONEY_THEME_LABELS: List[str] = ["Money In", "Money Out", "Money and relationships", "Thinking about money", "life in general"]

LIFE_CONTEXT_LABELS: List[str] = ["Sleep problems", "Medical appointment", "Medication", "Medical expense", "Physical Pain", "Illness", "Mental health", "Eating out/ordering", "Cooking at home", "Food struggle", "Work stress", "Job transition", "Job search", "Salary", "Wages", "Health coverage", "Unemployment", "Overtime", "Promotion", "Career change", "Side hustle", "School", "Studying", "Conflict", "School fees", "Student loans", "Socializing", "Caregiving", "Dating", "Intimacy", "Breakup", "Divorce", "Infidelity", "Children", "Parents", "In-laws", "Siblings", "Extended family", "Partner", "Friend", "Colleagues", "Classmates", "Peer pressure", "Family pressure", "Pets", "Meeting expectations", "Service providers", "Exercise", "Diet", "Fitness", "Progress", "Challenge", "Hobby", "Creativity", "Vacation/travel", "Entertainment", "Gaming", "Spiritual practice", "Spiritual crisis", "Substance use", "Alcohol use", "Bill arrived", "Payment Due", "Payment overdue", "Fines", "Money received", "Checking finances", "Impulse spending", "Known overspending", "Comfort spending", "Esteem spending", "Rebellion spending", "Convenience spending", "Online shopping", "Sale shopping", "Relational spending", "Financial avoiding", "Saving", "Investing", "Stock market change", "Gambling", "Borrowing from a bank", "Borrowing from a friend or family", "Lending to a friend", "Stealing", "Debt payment", "Collection call", "Can't afford something", "Taxes", "Legal matter", "Waiting", "Hard decision", "Big decision", "Rock and a Hard Place", "Housing", "Moving", "Relocation", "Transportation", "Vehicle", "Commute", "Social Media", "Scrolling", "Systemic failure", "Discrimination", "Institutional Failure", "Bureaucracy", "Red tape", "Passing the buck", "Hiding the truth", "Keeping Secrets", "Taboo", "Nature", "Weather", "City", "Achievement", "Celebration", "Mourning", "Grief", "Loss", "Embarrassing moment", "Emergency", "Crisis"]

STATUS_LABELS: List[str] = ["unmet", "met", "engaged"]

EMOTION_TO_FAMILY: Dict[str, str] = {"Dread": "Anxiety", "Anxiety": "Anxiety", "Nervous": "Anxiety", "Rage": "Anger", "Anger": "Anger", "Irritation": "Anger", "Humiliated Rage": "Anger", "Frustration": "Anger", "Resentment": "Anger", "Ruefulness": "Anger", "Fed Up": "Anger", "Bitterness": "Anger", "Mortification": "Shame", "Shame": "Shame", "Self-consciousness": "Shame", "Cringe": "Shame", "Guilt": "Guilt", "Regret": "Guilt", "Remorse": "Guilt", "Angry regret": "Guilt", "Despair": "Sadness", "Sadness": "Sadness", "Dismay": "Sadness", "Hurt": "Sadness", "Desperation": "Sadness", "Burnout": "Exhaustion", "Exhaustion": "Exhaustion", "Tiredness": "Exhaustion", "Overwhelm": "Exhaustion", "Panic": "Fear", "Fear": "Fear", "Unease": "Fear", "Vulnerable": "Vulnerable", "Insecure": "Vulnerable", "Helpless": "Vulnerable", "Shaky": "Vulnerable", "Envy": "Envy", "Jealousy": "Envy", "Revulsion": "Disgust", "Disgust": "Disgust", "Aversion": "Disgust", "Restlessness": "Boredom", "Boredom": "Boredom", "Indifference": "Boredom", "Elation": "Happiness", "Happiness": "Happiness", "Gladness": "Happiness", "Amusement": "Happiness", "Liberation": "Relief", "Relief": "Relief", "Awe": "Admiration", "Admiration": "Admiration", "Gratitude": "Admiration", "Triumph": "Pride", "Pride": "Pride", "Confidence": "Pride", "Serenity": "Calm", "Calm": "Calm", "Mellow": "Calm", "Shock": "Surprise", "Astonishment": "Surprise", "Surprise": "Surprise", "Nostalgia": "Bittersweet", "Longing": "Bittersweet", "Bittersweet": "Bittersweet", "Excitement": "Anticipation", "Optimism": "Anticipation", "Eagerness": "Anticipation", "Anticipation": "Anticipation", "Interest": "Anticipation", "Ambivalence": "Ambivalence", "Torn": "Ambivalence"}

NEED_TO_FAMILY: Dict[str, str] = {"Competence": "Competence", "Accomplishment": "Competence", "Effectiveness": "Competence", "Growth": "Competence", "Mastery": "Competence", "Strength": "Competence", "Impact": "Competence", "Hope": "Competence", "Progress": "Competence", "Challenge": "Competence", "Independence": "Autonomy", "Authenticity": "Autonomy", "Choice": "Autonomy", "Space": "Autonomy", "Boundaries": "Autonomy", "Freedom": "Autonomy", "Participation": "Autonomy", "Self-expression": "Autonomy", "Autonomy": "Autonomy", "Knowledge": "Discovery", "Creativity": "Discovery", "Adventure": "Discovery", "Discovery": "Discovery", "Stimulation": "Discovery", "Acceptance": "Connection", "Inclusion": "Connection", "Support": "Connection", "Companionship": "Connection", "Connection": "Connection", "Affection": "Connection", "Belonging": "Connection", "Community": "Connection", "Visibility": "Connection", "Love": "Connection", "Attention": "Connection", "Intimacy": "Connection", "Nurturance": "Connection", "Care": "Connection", "Trust": "Connection", "Order": "Security", "Security": "Security", "Predictability": "Security", "Physical safety": "Security", "Emotional safety": "Security", "Shelter": "Security", "Stability": "Security", "Protection": "Security", "Reassurance": "Security", "Ease": "Security", "Transparency": "Security", "Justice": "Fairness", "Accountability": "Fairness", "Social Justice": "Fairness", "Equity": "Fairness", "Relational Balance": "Fairness", "Fairness": "Fairness", "Equality": "Fairness", "Vindication": "Fairness", "Redemption": "Healing", "Replenishment": "Healing", "Healing": "Healing", "Forgiveness": "Healing", "Wholeness": "Healing", "Closure": "Healing", "Self-Respect": "Integrity", "Loyalty": "Integrity", "Self-Honesty": "Integrity", "Dignity": "Integrity", "Integrity": "Integrity", "Dependability": "Integrity", "Appreciation": "Understanding", "Recognition": "Understanding", "Respect": "Understanding", "Clarity": "Understanding", "Esteem": "Understanding", "Empathy": "Understanding", "Compassion": "Understanding", "Acknowledgement": "Understanding", "Validation": "Understanding", "Understanding": "Understanding", "Tranquility": "Peace", "Harmony": "Peace", "Solitude": "Peace", "Reflection": "Peace", "Beauty": "Peace", "Peace": "Peace", "Exertion": "Physical", "Rest": "Physical", "Physical": "Physical", "Pleasure": "Physical", "Nourishment": "Physical", "Touch": "Physical", "Vitality": "Physical", "Energy": "Physical", "Sexual expression": "Physical", "Humour": "Play", "Fun": "Play", "Play": "Play", "Laughter": "Play", "Spontaneity": "Play", "Leisure": "Play", "Engagement": "Purpose", "Mourning": "Purpose", "Awareness": "Purpose", "Inspiration": "Purpose", "Celebration": "Purpose", "Contribution": "Purpose", "Meaning": "Purpose", "Legacy": "Purpose", "Transcendence": "Purpose", "Generativity": "Purpose", "Purpose": "Purpose"}

EMOTION_FAMILY_VALENCE: Dict[str, str] = {"Anxiety": "negative", "Anger": "negative", "Shame": "negative", "Guilt": "negative", "Sadness": "negative", "Exhaustion": "negative", "Fear": "negative", "Vulnerable": "negative", "Envy": "negative", "Disgust": "negative", "Boredom": "negative", "Happiness": "positive", "Relief": "positive", "Admiration": "positive", "Pride": "positive", "Calm": "positive", "Surprise": "neutral", "Bittersweet": "neutral", "Anticipation": "neutral", "Ambivalence": "neutral"}

LIFE_CONTEXT_TO_FAMILY: Dict[str, str] = {
    "Sleep problems": "Sleep",
    "Medical appointment": "Health",
    "Medication": "Health",
    "Medical expense": "Health",
    "Physical Pain": "Health",
    "Illness": "Health",
    "Mental health": "Health",
    "Eating out/ordering": "Food",
    "Cooking at home": "Food",
    "Food struggle": "Food",
    "Work stress": "Work",
    "Job transition": "Work",
    "Job search": "Work",
    "Salary": "Work Details",
    "Wages": "Work Details",
    "Health coverage": "Work Details",
    "Unemployment": "Work Details",
    "Overtime": "Work Details",
    "Promotion": "Work Details",
    "Career change": "Work Details",
    "Side hustle": "Work Details",
    "School": "School / Education",
    "Studying": "School / Education",
    "Conflict": "Relational",
    "School fees": "Education Finance",
    "Student loans": "Education Finance",
    "Socializing": "Relational",
    "Caregiving": "Relational",
    "Dating": "Relational",
    "Intimacy": "Relational",
    "Breakup": "Relational",
    "Divorce": "Relational",
    "Infidelity": "Relational",
    "Children": "Relationships",
    "Parents": "Relationships",
    "In-laws": "Relationships",
    "Siblings": "Relationships",
    "Extended family": "Relationships",
    "Partner": "Relationships",
    "Friend": "Relationships",
    "Colleagues": "Relationships",
    "Classmates": "Relationships",
    "Peer pressure": "Social Pressure",
    "Family pressure": "Social Pressure",
    "Pets": "Pets",
    "Meeting expectations": "Social Pressure",
    "Service providers": "Relationships",
    "Exercise": "Exercise",
    "Diet": "Fitness / Wellbeing",
    "Fitness": "Fitness / Wellbeing",
    "Progress": "Fitness / Wellbeing",
    "Challenge": "Fitness / Wellbeing",
    "Hobby": "Recreation",
    "Creativity": "Recreation",
    "Vacation/travel": "Recreation",
    "Entertainment": "Recreation",
    "Gaming": "Recreation",
    "Spiritual practice": "Spiritual",
    "Spiritual crisis": "Spiritual",
    "Substance use": "Substance / Alcohol",
    "Alcohol use": "Substance / Alcohol",
    "Bill arrived": "Money Situations",
    "Payment Due": "Money Situations",
    "Payment overdue": "Money Situations",
    "Fines": "Money Situations",
    "Money received": "Money Situations",
    "Checking finances": "Money Situations",
    "Impulse spending": "Money Situations",
    "Known overspending": "Money Situations",
    "Comfort spending": "Money Situations",
    "Esteem spending": "Money Situations",
    "Rebellion spending": "Money Situations",
    "Convenience spending": "Money Situations",
    "Online shopping": "Money Situations",
    "Sale shopping": "Money Situations",
    "Relational spending": "Money Situations",
    "Financial avoiding": "Money Situations",
    "Saving": "Money Situations",
    "Investing": "Money Situations",
    "Stock market change": "Money Situations",
    "Gambling": "Money Situations",
    "Borrowing from a bank": "Money Situations",
    "Borrowing from a friend or family": "Money Situations",
    "Lending to a friend": "Money Situations",
    "Stealing": "Money Situations",
    "Debt payment": "Money Situations",
    "Collection call": "Money Situations",
    "Can't afford something": "Money Situations",
    "Taxes": "Money Situations",
    "Legal matter": "Money Situations",
    "Waiting": "Decisions / Limbo",
    "Hard decision": "Decisions / Limbo",
    "Big decision": "Decisions / Limbo",
    "Rock and a Hard Place": "Decisions / Limbo",
    "Housing": "Housing",
    "Moving": "Housing",
    "Relocation": "Housing",
    "Transportation": "Transportation",
    "Vehicle": "Transportation",
    "Commute": "Transportation",
    "Social Media": "Social Media / Scrolling",
    "Scrolling": "Social Media / Scrolling",
    "Systemic failure": "Systems / Institutions Failure / Bureaucracy",
    "Discrimination": "Systems / Institutions Failure / Bureaucracy",
    "Institutional Failure": "Systems / Institutions Failure / Bureaucracy",
    "Bureaucracy": "Systems / Institutions Failure / Bureaucracy",
    "Red tape": "Systems / Institutions Failure / Bureaucracy",
    "Passing the buck": "Systems / Institutions Failure / Bureaucracy",
    "Hiding the truth": "Hiding / Keeping Secret",
    "Keeping Secrets": "Hiding / Keeping Secret",
    "Taboo": "Hiding / Keeping Secret",
    "Nature": "Environment",
    "Weather": "Environment",
    "City": "Environment",
    "Achievement": "Life Events",
    "Celebration": "Life Events",
    "Mourning": "Life Events",
    "Grief": "Life Events",
    "Loss": "Life Events",
    "Embarrassing moment": "Life Events",
    "Emergency": "Life Events",
    "Crisis": "Life Events",
}


def _to_id(labels: List[str]) -> Dict[str, int]:
    return {l: i for i, l in enumerate(labels)}

def _to_label(labels: List[str]) -> Dict[int, str]:
    return {i: l for i, l in enumerate(labels)}


def _unique_in_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


emotion_to_id = _to_id(EMOTION_LABELS)
id_to_emotion = _to_label(EMOTION_LABELS)
need_to_id = _to_id(NEED_LABELS)
id_to_need = _to_label(NEED_LABELS)
money_theme_to_id = _to_id(MONEY_THEME_LABELS)
id_to_money_theme = _to_label(MONEY_THEME_LABELS)
life_context_to_id = _to_id(LIFE_CONTEXT_LABELS)
id_to_life_context = _to_label(LIFE_CONTEXT_LABELS)
status_to_id = _to_id(STATUS_LABELS)
id_to_status = _to_label(STATUS_LABELS)
EMOTION_FAMILY_LABELS = _unique_in_order([EMOTION_TO_FAMILY[label] for label in EMOTION_LABELS])
NEED_FAMILY_LABELS = _unique_in_order([NEED_TO_FAMILY[label] for label in NEED_LABELS])
LIFE_CONTEXT_FAMILY_LABELS = _unique_in_order([LIFE_CONTEXT_TO_FAMILY[label] for label in LIFE_CONTEXT_LABELS])
emotion_family_to_id = _to_id(EMOTION_FAMILY_LABELS)
id_to_emotion_family = _to_label(EMOTION_FAMILY_LABELS)
need_family_to_id = _to_id(NEED_FAMILY_LABELS)
id_to_need_family = _to_label(NEED_FAMILY_LABELS)
life_context_family_to_id = _to_id(LIFE_CONTEXT_FAMILY_LABELS)
id_to_life_context_family = _to_label(LIFE_CONTEXT_FAMILY_LABELS)

NUM_EMOTION = len(EMOTION_LABELS)
NUM_NEED = len(NEED_LABELS)
NUM_MONEY_THEME = len(MONEY_THEME_LABELS)
NUM_LIFE_CONTEXT = len(LIFE_CONTEXT_LABELS)
NUM_STATUS = len(STATUS_LABELS)
NUM_EMOTION_FAMILY = len(EMOTION_FAMILY_LABELS)
NUM_NEED_FAMILY = len(NEED_FAMILY_LABELS)
NUM_LIFE_CONTEXT_FAMILY = len(LIFE_CONTEXT_FAMILY_LABELS)


def emotion_family(emotion: str) -> str:
    return EMOTION_TO_FAMILY.get(emotion, "")


def emotion_valence(emotion: str) -> str:
    fam = emotion_family(emotion)
    return EMOTION_FAMILY_VALENCE.get(fam, "neutral")


def need_family(need: str) -> str:
    return NEED_TO_FAMILY.get(need, "")


def life_context_family(life_context: str) -> str:
    return LIFE_CONTEXT_TO_FAMILY.get(life_context, "")


def emotion_family_id(emotion: str) -> int:
    return emotion_family_to_id[emotion_family(emotion)]


def need_family_id(need: str) -> int:
    return need_family_to_id[need_family(need)]


def life_context_family_id(life_context: str) -> int:
    return life_context_family_to_id[life_context_family(life_context)]


def emotion_indices_for_family(family: str) -> List[int]:
    return [emotion_to_id[label] for label in EMOTION_LABELS if EMOTION_TO_FAMILY[label] == family]


def need_indices_for_family(family: str) -> List[int]:
    return [need_to_id[label] for label in NEED_LABELS if NEED_TO_FAMILY[label] == family]


def life_context_indices_for_family(family: str) -> List[int]:
    return [life_context_to_id[label] for label in LIFE_CONTEXT_LABELS if LIFE_CONTEXT_TO_FAMILY[label] == family]
