"""Default vocabulary for CLIP zero-shot moment classification.

These descriptive prompts are scored against a frame's CLIP embedding to surface
"what CLIP sees" in a moment, to describe your taste ("what you're into"), and to
drive auto-tagging. It's deliberately broad and caption-flavoured (CLIP was
trained on web captions, so natural noun-phrases score better than bare
keywords). Users override the whole list by dropping a `vocab.txt` (one prompt
per line) into /config — editable live from the GUI.

Kept descriptive and functional on purpose: this catalogues an adult-media
library, so the terms explicitly describe performers (build, hair, skin),
wardrobe, acts and positions. Add/trim freely — nothing assumes a fixed length.
"""

from __future__ import annotations

DEFAULT_VOCAB: list[str] = [
    # --- camera / framing ---
    "a close-up shot", "an extreme close-up", "a medium shot", "a wide shot",
    "a full body shot", "a point-of-view shot", "a POV blowjob", "an overhead angle",
    "a low angle looking up", "a mirror reflection", "a selfie", "a webcam view",
    "a handheld camera", "a professional studio shot", "an amateur home video",
    # --- setting / location ---
    "a bedroom", "a bathroom", "a shower", "a bathtub", "a kitchen",
    "a living room", "an office", "a hotel room", "a locker room", "a gym",
    "a swimming pool", "a hot tub", "a beach", "outdoors in nature", "a garden",
    "a car interior", "a public place", "a nightclub", "a stage", "a plain backdrop",
    # --- hair colour x length x style ---
    "a blonde woman", "long platinum blonde hair", "a brunette woman",
    "a woman with red hair", "a woman with black hair", "a woman with brown hair",
    "dyed or brightly colored hair", "pink hair", "long hair", "short hair",
    "shoulder length hair", "a ponytail", "pigtails", "braided hair",
    "curly hair", "straight hair", "wavy hair", "a bob haircut", "wet hair",
    "hair tied up in a bun",
    # --- eyes / face ---
    "blue eyes", "brown eyes", "green eyes", "hazel eyes",
    "heavy eye makeup", "red lipstick", "freckles", "wearing glasses",
    # --- skin tone ---
    "fair pale skin", "light skin", "olive skin", "tan skin", "brown skin",
    "dark skin", "ebony skin", "a black woman", "an asian woman", "a latina woman",
    "a white woman", "sun-tanned skin", "oiled shiny skin",
    # --- body type / build (full range) ---
    "a slim woman", "a skinny woman", "a petite woman", "a small woman",
    "an athletic fit woman", "a toned body", "a curvy woman", "an hourglass figure",
    "a voluptuous woman", "a thick woman", "a chubby woman", "a plus-size BBW woman",
    "a fat woman", "a big beautiful woman", "wide hips", "a small waist",
    "a flat stomach", "a pregnant woman", "a tall woman", "a muscular woman",
    # --- breasts ---
    "small breasts", "medium breasts", "large breasts", "huge breasts",
    "enormous breasts", "natural breasts", "a woman with big natural breasts",
    "enhanced fake breasts", "perky breasts", "saggy breasts", "cleavage",
    "bare breasts", "puffy nipples", "pierced nipples",
    # --- butt ---
    "a small butt", "a round butt", "a big butt", "an enormous butt",
    "a bubble butt", "a thong framing the butt", "a spread butt",
    # --- other body detail ---
    "a person with tattoos", "a person with piercings", "a belly button piercing",
    "a shaved pussy", "a hairy pussy", "a trimmed bush", "long painted nails",
    "a petite young woman", "a mature woman", "a milf",
    # --- wardrobe: type ---
    "wearing lingerie", "wearing a bra and panties", "wearing a thong",
    "wearing a g-string", "wearing a corset", "wearing a babydoll",
    "wearing a teddy", "wearing a lace bra", "wearing a garter belt",
    "wearing a dress", "wearing a tight dress", "wearing a skirt",
    "wearing a miniskirt", "wearing a bikini", "wearing a microbikini",
    "wearing a swimsuit", "wearing a one-piece swimsuit", "wearing jeans",
    "wearing shorts", "wearing yoga pants", "wearing leggings", "wearing a crop top",
    "wearing a t-shirt", "wearing a bodysuit", "wearing a uniform",
    "wearing a schoolgirl outfit", "wearing a maid outfit", "wearing a nurse costume",
    "wearing a costume", "wearing a bunny outfit", "wearing latex",
    "wearing leather", "wearing high heels", "wearing boots", "wearing thigh-high boots",
    "wearing stockings", "wearing fishnet stockings", "wearing pantyhose",
    "wearing socks", "wearing a collar", "wearing a blindfold", "wearing handcuffs",
    "wearing a choker", "wearing jewelry",
    # --- wardrobe: state ---
    "topless", "bottomless", "fully nude", "partially clothed",
    "clothes pulled aside", "a skirt lifted up", "panties pulled down",
    "panties pulled aside", "a bra removed", "undressing", "wet clothing",
    "wearing casual clothes", "see-through sheer clothing",
    # --- wardrobe: colour ---
    "red lingerie", "black lingerie", "white lingerie", "pink lingerie",
    "black stockings", "white stockings", "red high heels", "white panties",
    "black panties", "a red dress", "a black dress", "a white dress",
    "a black corset", "a white bikini", "a black bikini",
    # --- people / configuration ---
    "one woman alone", "a solo woman", "two women together", "a man and a woman",
    "a couple having sex", "a threesome", "a group of people", "a gangbang",
    "two people together", "a lesbian couple", "an interracial couple",
    # --- pose / non-explicit action ---
    "standing", "sitting", "kneeling", "lying on a bed", "lying on her back",
    "bending over", "on hands and knees", "arching the back", "spreading the legs",
    "legs up in the air", "squatting", "straddling", "dancing", "twerking",
    "stripping off clothes", "posing for the camera", "stretching",
    "touching herself", "masturbating", "fingering herself", "using a vibrator",
    "using a dildo", "kissing", "an embrace", "a massage", "spanking",
    # --- explicit acts ---
    "a woman performing oral sex", "a blowjob", "deepthroat", "licking a penis",
    "cunnilingus", "eating pussy", "vaginal penetration", "anal penetration",
    "a handjob", "a titjob", "a footjob", "penetration close-up",
    "double penetration", "a facial cumshot", "cum on the face", "cum on the body",
    "creampie", "a woman swallowing", "a cumshot", "69 position",
    "rimming", "fisting", "squirting",
    # --- positions ---
    "the missionary position", "the doggy style position", "the cowgirl position",
    "reverse cowgirl", "riding on top", "the spooning position",
    "legs on shoulders", "standing sex", "sex from behind", "pinned against a wall",
    # --- facial expression / mood ---
    "a look of pleasure", "an open mouth moaning", "eye contact with the camera",
    "biting her lip", "a seductive expression", "a shy expression",
    "sticking out her tongue", "closed eyes in ecstasy",
    # --- lighting / mood / style ---
    "dim mood lighting", "bright lighting", "natural daylight", "neon lighting",
    "candlelight", "a colorful background", "a dark background",
    "a high quality photo", "a grainy low quality photo", "black and white",
]
