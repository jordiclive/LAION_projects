import os
import time

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

model_name = "jordiclive/flan-t5-3b-summarizer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
kwargs = dict(device_map="balanced_low_0", torch_dtype=torch.bfloat16)

t_start = time.time()
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()
target_length = 150
max_source_length = 512

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
prompts = {
    "article": "Produce an article summary of the following news article:",
    "one_sentence": "Given the following news article, summarize the article in one sentence:",
    "conversation": "Briefly summarize in third person the following conversation:",
    "scitldr": "Given the following scientific article, provide a TL;DR summary:",
    "bill": "Summarize the following proposed legislation (bill):",
    "outlines": "Produce an article summary including outlines of each paragraph of the following article:",
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["TRANSFORMERS_CACHE"] = "/admin/home-jordiclive/transformers_cache"


def generate(inputs, max_source_length=512, summarization_type=None, prompt=None):
    """returns a list of zipped inputs, outputs and number of new tokens"""

    if prompt is not None:
        inputs = [f"{prompt.strip()} {i.strip()}" for i in inputs]

    if summarization_type is not None:
        inputs = [f"{prompts[summarization_type].strip()} {i.strip()}" for i in inputs]
    if summarization_type is None and prompt is None:
        inputs = [f"Summarize the following: {i.strip()}" for i in inputs]
    input_tokens = tokenizer.batch_encode_plus(
        inputs,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    outputs = model.generate(
        **input_tokens,
        use_cache=True,
        num_beams=5,
        min_length=5,
        max_new_tokens=target_length,
        no_repeat_ngram_size=3,
    )

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return inputs, outputs, total_new_tokens

import torch
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model_name,
torch_dtype=torch.bfloat16
)

wall_of_text = "your words here"

def generate(inputs, max_source_length=512, summarization_type=None, prompt=None):
    """returns a list of zipped inputs, outputs and number of new tokens"""

    if prompt is not None:
        inputs = [f"{prompt.strip()} {i.strip()}" for i in inputs]

    if summarization_type is not None:
        inputs = [f"{prompts[summarization_type].strip()} {i.strip()}" for i in inputs]
    if summarization_type is None and prompt is None:
        inputs = [f"Summarize the following: {i.strip()}" for i in inputs]

    result = summarizer(
        inputs[0],
        num_beams=5,
        min_length=5,
        no_repeat_ngram_size=3,
    )
    return 1, result, 1

inputs = [
    """A plum, which was 'accidentally created' , has been touted as the next superfood to rival the acai berry. The¬†Australian Queen Garnet contains some of the highest levels of antioxidants ever found in a fruit and has just gone on sale in the UK. According to studies the fruit has five to ten times more anthocyanins than a normal plum. The Australian Queen Garnet has five to ten times more of the antioxidant¬†anthocyanins than a normal plum . The superplum was created during a breeding programme for a disease-resistant version of the common plum . Anthocyanins provide the dark colours of many fruits and vegetables, such as blueberries and red peppers. Research suggests that the plant antioxidants, which mop up harmful molecules, can help protect arteries and prevent the DNA damage that leads to cancer. The plum was accidentally created during a breeding programme for a disease-resistant version of the common plum in Queensland, Australia. It is currently undergoing trials in studies with obese rats, and early results relating to the fruit's potential to aid weight loss have been positive, reports say. Rats were fed a diet high in fats and carbohydrates until they were obese. Then a few drops of the plum juice was added to the rodent's drinking water, which the rats consumed during their daily 30 minutes of exercise. The previously obese rats shed most of their excess weight in eight weeks. Anthocyanins provide the dark colours of many fruits and vegetables, such as blueberries and red peppers. They have high levels of antioxidants which appeare to reduce inflammation and reduce the effects of arthritis in the body. The compounds in the antioxidants  have similar effects as drugs such as aspirin and ibuprofen. Professor Lindsay Brown, ¬†from USQ Biomedical Sciences who led the research told the BBC that the results from the rodent-led research are enough to make a case for the plums health claims. 'All the changes that rats experience with obesity - glucose levels, cardiovascular functions, inflammation - all those occur the same way in humans,' Prof Brown said, adding: 'The plums taste really nice.' Marks & Spencer claims it is the first retailer in the UK to stock the new variety of plum. M&S fruit technologist Andrew Mellonie said: 'This is one of the most exciting new fruits to hit the UK in the last decade. 'It is a win-win fruit. Not only is it delicious but also incredibly healthy. It was created purely by chance and is an amazing discovery that could offer significant health benefits.' Similar levels of anthocyanins in berries usually made them inedible, but the Queen Garnet is incredibly sweet and a 'delicious addition to the fruit bowl' with a 'delicious jammy taste - similar to that of a black fig', Mr Mellonie continued . But not everyone agrees with the positive claims. The plum can be consumed as a juice (left) or stewed and eaten with ice-cream for pudding (right) Professor Manny Noakes, research director for nutrition and health at the Australian science agency CSIRO, said it wasn't a clear-cut case. 'It's very good research and very interesting research,' ¬†Prof Noakes said to the BBC. 'But when I last checked, humans and rodents were very different. You can feed rats an entire diet to test a hypothesis but that doesn't mean you'll get the same results in humans. 'To make a claim that the consumption of a food will make a difference to people's weight is a pretty long bow to cast,' she continued. 'Unfortunately, this is something that happens a lot when it comes to promoting the health benefits of food. Similar claims have been made by research on animals using everyday grape seeds.' Mr Mellonie said: 'We're constantly working with our growers around the world to develop new fruits and vegetables for our customers to try. 'When our supplier came to us with the Queen Garnet we knew we had to get it on our shelves for consumers in the UK to enjoy.' The plums are currently being grown for M&S in Australia, although the retailer said it hoped to grow them closer to the UK in future. They have gone on sale for ¬£3 for 400g,¬†or two packs for ¬£4. Moringa . Best for: Liver protection and blood sugar regulation . Most popular in powder form, this superfood is made from the leaf of the moringa ‚Äòmiracle' tree, native to Africa and Asia and one of the most nutrient-dense plants on the planet. It has been claimed that it boosts immunity, lowers blood pressure, alleviates stress, fights fatigue, improves digestive health and increases libido. And that's just on the inside. It's also a tonic for hair, nails and skin. So what makes moringa so miraculous? It contains almost 25 per cent protein, including all nine of the essential amino acids, which are important for the body's key functions and help maintain healthy skin cells. It is also packed with essential vitamins and minerals, including skin-supporting vitamins such as zinc, which includes strong, supple healthy skin, nails and hair among its beauty benefits. Holistic nutritionist Nikki Baker advises: 'Moringa has six times the antioxidant content of the popular superfood, goji berries. It contains over 90 nutrients, 46 antioxidants and abundant minerals. 'Gram for gram, Moringa contains more vitamin B12 than steak, more vitamin A than eggs, and more calcium than milk. 'Its antioxidants detox and protect the liver, boost immunity and support healthy blood sugar levels.' Chlorella . Best for: Liver detoxification and healthy protein delivery . We‚Äôve all seen supermodels sipping on green juices containing chlorella this year but as Nikki stresses, there‚Äôs more to this trend than just a fad, especially when it comes to the morning after. The one to look out for, she says, is Sun Chlorella A, a green algae and superfood detoxifier, popular in Japan. In 1996, research from Sapporo Medical University, Japan, suggested that chlorella, a freshwater green algae, can lessen hangover symptoms by up to 96 per cent if taken before drinking. The idea is that the antioxidant-packed algae neutralises the free radicals (chemicals that damage cells) in alcohol. Chlorella - which also contains iron, folic acid and energy-boosting B vitamins - is also lauded for its gut-soothing properties, as it dissolves and expands to coat the stomach when eaten. Mushrooms . Best for: Adrenal recovery and digestion . These aren‚Äôt the kind of fungi you‚Äôll find in the veg section of the supermarket. These medicinal mushrooms have powerful antioxidant properties that can help bring you back on track following a heavy night. One such example is chaga, a hard black fungus which has been central to folk medicine in northern climates for centuries and is dubbed the mushroom of immortality in Siberia. Nikki said: 'Chaga is a nutrient-rich medicinal mushroom, it grows on the birch tree and so harnesses the amazing nutrient properties of this tree. 'Chaga contains numerous B vitamins, flavonoids, minerals and enzymes. 'It is also one of the world‚Äôs densest sources of pantothenic acid, an essential nutrient needed by the adrenal glands as well as digestive organs."""
]
_, outputs, _ = generate(inputs, summarization_type="article")
print(outputs)

_, outputs, _ = generate(inputs, summarization_type="one_sentence")
print(outputs)
inputs = [
    """How was it? :D
Carrie: Or is it still going? ;>
Olivia: yes
Carrie: whoah, good!
Olivia: no
Olivia: 30
Carrie: Waiting for more, you got me curious
Olivia: Jesus, I thought it'll never end
Carrie: That bad?
Olivia: It wasn't awful if that's what you're asking, but... I don't know, I didn't feel anything?
Carrie: I understand, it happens
Carrie: But will meet again?
Olivia: He texted me already and wants to meet
Carrie: He must have liked you then! That's good
Olivia: No if I don't like him... And I kind of think I don't. He was nice and all, but...
Carrie: You don't fancy him?
Olivia: :( it's so shallow... I feel really bad about it
Carrie: I think you shouldn't - what guy would say the same? We both know they wouldn't talk to a girl they don't find pretty so don't beat yourself about it
Olivia: Thanks dear
Olivia: What should I do? He asked me out again
Carrie: I‚Äôd give him another chance. Sometimes there are no sparkles in the beginning
Carrie: But you said you like talking to him
Olivia: Yeah, he‚Äôs nice, I‚Äôm not sure if not too nice ;)
Carrie: Hahaha, we sure are from the same family
Olivia: I said I‚Äôd go out with him again
Olivia: Ok, he suggested we can go to the cinema again‚Ä¶
Carrie: Very creative indeed :D
Olivia: Maybe I‚Äôm weird, but come on, I don‚Äôt feel like going any more.
Carrie: Eh‚Ä¶ Go out with him once more and if you‚Äôre bored again just tell him no."""
]
_, outputs, _ = generate(inputs, summarization_type="conversation")
print(outputs)

_, outputs, _ = generate(
    inputs,
    summarization_type=None,
    prompt="Summarize the conversation in less than 8 words",
)
print(outputs)

inputs = [
    """You must be 18 years old, live or work in New York State, permanent resident alien status and have no recent felony convictions. If you don't meet these requirements, your application will be denied. There are no special education requirements.;
, This can be obtained from the New York Secretary of State, Division of Licensing services; or get it from the New York State Notary Public Association by calling 1-877-484-4673.¬†


Do not worry.¬†The exam is only 40-multiple choice questions and you only need to score 70%.¬† (That means you can even miss 12 and still pass.)

,¬†There is a 60 percent failure rate for people who simply walk into the exam‚Äì‚Äìmore than likely this is due to them not studying.People who take an online class or attend the 5-hour class with the N.Y.S. Notary Association pass with a 99 percent rate.Membership is free with the association.
¬†You do not need a job to qualify. In fact, this credential will certainly help you get a job if you are looking for a resume boost. The exam is given on a weekly basis in every major metropolitan area throughout N.Y. State except on state holidays.
 Bring a government issued photo ID that is not expired and has your signature.¬†A driver's license is perfect.¬†Bring a check or money order for $15, payable to:¬† "Secretary of State".¬†You can also use a Visa/Mastercard debit/credit card at the door.¬†


You will also be thumb printed.¬†
Latecomers are not admitted into the exam.
 It will be accompanied by your "Oath of Office" affidavit.
, Include a $60 check made out to the "Secretary Of State".¬†Congratulations, your Notary Public license will arrive in the mail within 6 to 8 weeks."""
]

_, outputs, _ = generate(
    inputs,
    summarization_type=None,
)
print(outputs)
_, outputs, _ = generate(
    inputs,
    summarization_type='one_sentence'
)
print(outputs)

_, outputs, _ = generate(
    inputs,
    summarization_type=None,
    prompt="Summarize the conversation in less than 8 words",
)
print(outputs)

