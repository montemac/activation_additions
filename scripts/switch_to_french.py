""" This script demonstrates a "switch to speaking French" vector, based
off of user faul_sname's code (https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector?commentId=sqsS9QaDy2bG83XKP). """

# %%
from typing import List

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import (
    completion_utils,
    utils,
    hook_utils,
    prompt_utils,
)
from activation_additions.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

utils.enable_ipython_reload()

# %%
_ = torch.set_grad_enabled(False)
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda:2")

# %% Check that the model can speak French at all
french_prompt = (
    "Il est devenu maire en 1957 après la mort d'Albert Cobo et a été élu à"
    " part entière peu de temps après par une marge de 6: 1 sur son"
    " adversaire. Miriani était surtout connue pour avoir réalisé de nombreux"
    " projets de rénovation urbaine à grande échelle initiés par"
    " l'administration Cobo et largement financés par des fonds fédéraux."
    " Miriani a également pris des mesures énergiques pour surmonter le taux"
    " de criminalité croissant à Detroit."
)

completion_utils.print_n_comparisons(
    prompt=french_prompt,
    num_comparisons=3,
    model=model,
    activation_additions=[],
    seed=0,
    tokens_to_generate=60,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)  # GPT2-XL basically can't speak French properly

# %%
sentence_pairs = [
    [
        (
            "The album| received| mixed to positive reviews,| with critics"
            " commending| the production| of many of the songs| while"
            " comparing| the album| to the electropop stylings| of Ke\$ha and"
            " Robyn."
        ),
        (
            "L'album| a reçu| des critiques mitigées à positives,| les"
            " critiques louant| la production| de nombreuses chansons| tout en"
            " comparant| l'album| aux styles électropop| de Ke\$ha et Robyn."
        ),
    ],
    [
        (
            "The river's flow| is the greatest| during| the snow melt season|"
            " from March to April,| the rainy season| from June to July| and"
            " during the typhoon season| from September to October."
        ),
        (
            "Le débit de la rivière| est le plus élevé| pendant| la saison de"
            " fonte des neiges| de mars à avril,| la saison des pluies| de"
            " juin à juillet| et pendant la saison des typhons| de septembre à"
            " octobre."
        ),
    ],
    [
        (
            "By law,| the Code Reviser| must be a lawyer;| however,| the"
            " functions| of the office| can also be delegated| by the Statute"
            " Law Committee| to a private legal publisher."
        ),
        (
            "Selon la loi,| le réviseur du code| doit être un avocat;|"
            " cependant,| les fonctions| du bureau| peuvent également être"
            " déléguées| par le Comité des lois statutaires| à un éditeur"
            " juridique privé."
        ),
    ],
]
activation_additions = []
coeff: float = 3
for sentence_en, sentence_fr in sentence_pairs:
    phrase_pairs = list(
        zip(*[s.split("|") for s in (sentence_en, sentence_fr)])
    )
    sentence_en = "".join(phrase_en for phrase_en, phrase_fr in phrase_pairs)
    print(sentence_en)
    for j in range(len(phrase_pairs) - 1, -1, -1):
        sentence_en2fr = "".join(
            pair[i >= j] for i, pair in enumerate(phrase_pairs)
        )
        print(sentence_en2fr)
        ave_en2fr_pos, ave_en_neg = prompt_utils.get_x_vector(
            prompt1=sentence_en2fr,
            prompt2=sentence_en,
            coeff=coeff / 56,  # 56 activation additions TODO avoid hardcoding
            act_name=24,
            model=model,
            pad_method="tokens_right",
        )
        activation_additions += [ave_en2fr_pos, ave_en_neg]
# %%
prompt = (
    "He became Mayor in 1957 after the death of Albert Cobo, and was elected"
    " in his own right shortly afterward by a 6:1 margin over his opponent."
    " Miriani was best known for completing many of the large-scale urban"
    " renewal projects initiated by the Cobo administration, and largely"
    " financed by federal money. Miriani also took strong measures to overcome"
    " the growing crime rate in Detroit."
)
completion_utils.print_n_comparisons(
    prompt=prompt,
    num_comparisons=3,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    tokens_to_generate=100,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
# %%
