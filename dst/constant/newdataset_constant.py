INTENT2IDX = {"greeting": 0,
              "req_temperature": 1,
              "thank": 2,
              "req_instruction": 3,
              "confirm": 4,
              "req_repeat": 5,
              "negate": 6,
              "req_amount": 7,
              "req_ingredient": 8,
              "other": 9,
              "req_is_recipe_finished": 10,
              "req_tool": 11,
              "req_duration": 12,
              "affirm": 13,
              "goodbye": 14,
              "req_substitute": 15,
              "req_confirmation": 16,
              "req_description": 17,
              "req_explanation": 18}

INTENTS = ["greeting",
           "req_temperature",
           "thank",
           "req_instruction",
           "confirm",
           "req_repeat",
           "negate",
           "req_amount",
           "req_ingredient",
           "other",
           "req_is_recipe_finished",
           "req_tool",
           "req_duration",
           "affirm",
           "goodbye",
           "req_substitute",
           "req_confirmation",
           "req_description",
           "req_explanation"]

INTENT_DESCRIPTIONS = {
    "greeting": "greeting",
    "req_temperature": "ask about the cooking temperature",
    "thank": "thank",
    "req_instruction": "ask for instructions",
    "confirm": "confirm the current stage",
    "req_repeat": "ask to repeat the last information",
    "negate": "negate",
    "req_amount": "ask about the amount information",
    "req_ingredient": "ask about the ingredients",
    "other": "other intent",
    "req_is_recipe_finished": "ask whether the recipe is finished",
    "req_tool": "ask about the cooking tool",
    "req_duration": "ask about cooking duration",
    "affirm": "affirm",
    "goodbye": "goodbye",
    "req_substitute": "ask for tool or ingredient substitutions",
    "req_confirmation": "ask for verification",
    "req_description": "ask for the description",
    "req_explanation": "ask to explain the reason or explain in more detail"
}


INTENT_DESCRIPTIONS2 = {
    "greeting": "greet to start a conversation",
    "req_temperature": "request the exact temperature expression for an action",
    "thank": "express gratitude",
    "req_instruction": "request the next instruction after having confirmed accomplishment of previous instructions",
    "confirm": "establish the fact that one or several actions have been accomplished",
    "req_repeat": "ask for repeating a mentioned entity or instruction",
    "negate": "give a negative statement",
    "req_amount": "ask for the exact quantity of an entity",
    "req_ingredient": "ask for the next ingredient",
    "other": "other intent",
    "req_is_recipe_finished": "ask if the cooking process ends or not",
    "req_tool": "request a specific tool entity",
    "req_duration": "request the time needed for an action",
    "affirm": "give a positive statement",
    "goodbye": "express good wishes at the end of a conversation",
    "req_substitute": "want to know if it is possible to use an alternative ingredient instead of the prescribed one",
    "req_confirmation": "ask for confirmation",
    "req_description": "ask to describe the change of ingredients",
    "req_explanation": "ask to explain the reason or explain in more detail"
}
