import json

data = {
    "focus": [
        "blurry",
        "out-of-focus",
        "soft-focus"
    ],

    "appearance": [
        "plain",
        "uniform",
        "textureless",
        "low-texture",
        "solid color",
        "uncluttered",
        "matte",
        "glossy",
        "reflective"
    ],

    "material": [
        "wooden",
        "plywood",
        "concrete",
        "cement",
        "stone",
        "brick",
        "stainless steel",
        "steel",
        "painted metal",
        "sheet metal",
        "aluminum",
        "galvanized metal",
        "rubber",
        "plastic",
        "acrylic",
        "glass",
        "ceramic",
        "tile",
        "fabric",
        "foam",
        "cardboard",
        "paper"
    ],

    "texture": [
        "smooth",
        "rough",
        "brushed",
        "polished",
        "corrugated",
        "ribbed",
        "perforated",
        "grainy",
        "patterned",
        "weathered",
        "worn"
    ],

    "context": [
        "surface",
        "tabletop",
        "work surface",
        "workbench",
        "worktable",
        "assembly table",
        "inspection table",
        "floor",
        "factory floor",
        "warehouse floor",
        "wall",
        "industrial wall",
        "panel",
        "metal panel",
        "machine surface",
        "machine enclosure",
        "equipment housing",
        "conveyor belt",
        "shelf",
        "rack",
        "cabinet interior",
        "pallet",
        "background",
        "backdrop"
    ]
}


prompts = []


# --------------------------------------------------
# 1. Material-based prompts
# --------------------------------------------------
for material in data["material"]:
    prompts.append(f"a photo of a {material} surface.")
    prompts.append(f"a photo of a {material} background.")


# --------------------------------------------------
# 2. Context-based prompts
# --------------------------------------------------
for context in data["context"]:
    prompts.append(f"a photo of a {context}.")
    prompts.append(f"a photo of an industrial {context}.")


# --------------------------------------------------
# 3. Focus-based prompts
# --------------------------------------------------
for focus in data["focus"]:
    prompts.append(f"a photo of a {focus} background.")
    prompts.append(f"a photo of a {focus} surface.")


# --------------------------------------------------
# 4. Appearance-based prompts
# --------------------------------------------------
for appearance in data["appearance"]:
    prompts.append(f"a photo of a {appearance} background.")
    prompts.append(f"a photo of a {appearance} surface.")


# --------------------------------------------------
# 5. Texture-based prompts
# --------------------------------------------------
for texture in data["texture"]:
    prompts.append(f"a photo of a {texture} surface.")


# --------------------------------------------------
# 6. Material + context
#    「意味的に自然な組み合わせ」を手動で定義
# --------------------------------------------------
material_context_pairs = [
    ("wooden", "workbench"),
    ("wooden", "worktable"),
    ("plywood", "work surface"),

    ("concrete", "floor"),
    ("concrete", "factory floor"),
    ("concrete", "warehouse floor"),

    ("cement", "floor"),
    ("stone", "floor"),
    ("brick", "wall"),

    ("stainless steel", "surface"),
    ("stainless steel", "work surface"),
    ("stainless steel", "tabletop"),

    ("steel", "surface"),
    ("steel", "panel"),
    ("steel", "machine enclosure"),

    ("painted metal", "panel"),
    ("painted metal", "machine surface"),
    ("painted metal", "equipment housing"),

    ("sheet metal", "panel"),
    ("sheet metal", "machine enclosure"),

    ("aluminum", "surface"),
    ("aluminum", "panel"),

    ("galvanized metal", "wall"),
    ("galvanized metal", "panel"),

    ("rubber", "surface"),
    ("rubber", "floor"),
    ("rubber", "conveyor belt"),

    ("plastic", "surface"),
    ("plastic", "machine enclosure"),
    ("plastic", "equipment housing"),

    ("acrylic", "panel"),
    ("glass", "surface"),
    ("glass", "panel"),

    ("ceramic", "surface"),
    ("tile", "floor"),
    ("tile", "wall"),

    ("fabric", "surface"),
    ("foam", "surface"),
    ("cardboard", "surface"),
    ("paper", "surface"),
]

for material, context in material_context_pairs:
    prompts.append(f"a photo of a {material} {context}.")


# --------------------------------------------------
# 7. Appearance + material
# --------------------------------------------------
appearance_material_pairs = [
    ("matte", "stainless steel"),
    ("matte", "painted metal"),
    ("matte", "plastic"),
    ("matte", "surface"),

    ("glossy", "plastic"),
    ("glossy", "painted metal"),
    ("glossy", "surface"),

    ("reflective", "stainless steel"),
    ("reflective", "metal"),
    ("reflective", "glass"),

    ("smooth", "concrete"),
    ("smooth", "metal"),
    ("smooth", "plastic"),
    ("smooth", "glass"),

    ("rough", "concrete"),
    ("rough", "stone"),
    ("rough", "cement"),

    ("brushed", "steel"),
    ("brushed", "stainless steel"),
    ("brushed", "aluminum"),

    ("polished", "steel"),
    ("polished", "stainless steel"),
    ("polished", "metal"),

    ("corrugated", "metal"),
    ("corrugated", "plastic"),

    ("weathered", "metal"),
    ("weathered", "concrete"),

    ("worn", "metal"),
    ("worn", "rubber"),
]

for appearance, material in appearance_material_pairs:
    prompts.append(f"a photo of a {appearance} {material} surface.")


# --------------------------------------------------
# 8. Focus + context
# --------------------------------------------------
focus_context_pairs = [
    ("blurry", "factory floor"),
    ("blurry", "warehouse floor"),
    ("blurry", "workbench"),
    ("blurry", "machine surface"),
    ("blurry", "background"),

    ("out-of-focus", "factory floor"),
    ("out-of-focus", "warehouse floor"),
    ("out-of-focus", "workbench"),
    ("out-of-focus", "machine surface"),
    ("out-of-focus", "background"),

    ("soft-focus", "factory floor"),
    ("soft-focus", "warehouse floor"),
    ("soft-focus", "workbench"),
    ("soft-focus", "background"),
]

for focus, context in focus_context_pairs:
    prompts.append(f"a photo of a {focus} {context}.")


# --------------------------------------------------
# 9. Appearance + context
# --------------------------------------------------
appearance_context_pairs = [
    ("plain", "background"),
    ("plain", "surface"),
    ("uniform", "background"),
    ("uniform", "surface"),
    ("textureless", "background"),
    ("textureless", "surface"),
    ("low-texture", "background"),
    ("low-texture", "surface"),
    ("solid color", "background"),
    ("uncluttered", "background"),
    ("uncluttered", "surface"),

    ("matte", "panel"),
    ("matte", "wall"),
    ("glossy", "panel"),
    ("reflective", "panel"),
]

for appearance, context in appearance_context_pairs:
    prompts.append(f"a photo of a {appearance} {context}.")


# --------------------------------------------------
# 10. Remove duplicates
# --------------------------------------------------
prompts = list(dict.fromkeys(prompts))


# --------------------------------------------------
# 11. Save to JSON
# --------------------------------------------------
output = {
    "background_prompts": prompts
}

with open("background_prompts.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)


# --------------------------------------------------
# 12. Display
# --------------------------------------------------
print(f"Number of prompts: {len(prompts)}")

for i, prompt in enumerate(prompts):
    print(f"{i:03d}: {prompt}")