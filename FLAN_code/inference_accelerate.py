import argparse
import gc
import math
import os
import time

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--name", type=str, help="Name path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
    parser.add_argument("--ckpt_path", type=str, help="float16 or int8")

    return parser.parse_args()
python inference_accelerate.py --name 'google/flan-t5-xl' --ckpt_path checkpoint-curr-best_20221125_1232 --benchmark


t_start = time.time()

num_tokens = 100

args = get_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()

rank = local_rank


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


print_rank0(f"Using {world_size} gpus")
model_name = args.name
print_rank0(f"Loading model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.bfloat16

# print(get_max_memory_per_gpu_dict())

# infer_dtype = args.dtype
# if infer_dtype == "int8":
#     dtype = torch.int8

kwargs = dict(
    device_map="balanced_low_0",
)

# if infer_dtype == "int8":
#     print_rank0("Using `load_in_8bit=True` to use quanitized model")
#     kwargs["load_in_8bit"] = True
# else:
#     kwargs["torch_dtype"] = dtype

model = 'google/flan-t5-xxl'
model = args.ckpt_path
model = AutoModelForSeq2SeqLM.from_pretrained(model, **kwargs)
# model.load_state_dict(torch.load('new_model.pt'))


if args.benchmark:
    t_ready = time.time()


### Generate

print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]

input_sentences = ['A car\'s fuel consumption is a measure of miles driven per gallon of gas. If you know the distance you drove and how many gallons fit in your tank, you can simply divide the miles by the gas to get your "miles per gallon," or mpg.\n\n\nYou can perform the same calculation with kilometers and liters as well.\nThe best time to record is right after you fill your car with gas.;\n, Newer cars have a trip odometer that you can set to zero at any time. It is usually on the dashboard or center console, with a small button you can hold to reset it to zero. Set it to zero when you fill up the car and check it when you need to fill up again -- this is you mileage since you last bought gas.\n\n\nYour trip odometer will say "0 miles."\nIf you don\'t have a trip odometer, record the number of miles on your car as "Starting Mileage." For example, if your car has 10,000 miles on it when you fill your tank, write "10,000."\n\n, Before you start filling up your car at the gas station, record the mileage on the odometer as "Final Mileage."\n\n\nIf you do not have a trip odometer, subtract your "Starting Mileage" from your current mileage to find out how far your traveled. If your odometer now says 10,250 for example, subtract 10,000. You drove 250 miles on that tank of gas.\n You can perform this calculation no matter how much gas is left in the tank, but the more gas you use the more accurate your reading will be.\n Refill your tank completely and note how many gallons/liters you needed to fill the tank back up. This is you "Fuel Usage."\n\n\nYou must refill your tank completely for this to work, otherwise you don\'t know how much gas your car used since your last tank.\n This tells you how many miles you drove per gallon of gas. For example, if you drove 335 miles before refueling, and you filled your car up with 12 gallons of gas, your fuel consumption was 27.9 miles per gallon, or mpg (335 miles / 12 gallon = 27.9 mpg).\n\n\nIf you measured in kilometers and liters your answer will be in kilometers per liter (kpl). Many Europeans multiply this answer by 100 to get "kilometers per 100 liters" of gas.\nYou have to start from a full tank and return to a full tank to know exactly how much gas your car consumed.\n Terry\'s odometer reads 23,500 with a full tank. After driving for a few days he needs to buy gas. The odometer reads 23,889, and it takes 12.5 gallons to refill his tank. What was his fuel consumption?\n\n\nFuel Consumption = (Final Mileage - Starting Mileage) / Fuel Usage\nFuel Consumption = (23,889mi - 23,500mi) / 12.5 gallons\nFuel Consumption = 389mi / 12.5 gallons\nFuel Consumption = 31.1 mpg', "If you have a great idea already, then build off of that. Have an idea before you write the song; it will make it a lot easier!;\n, Look up verses that resonate most with you. Can you see ways in which these will form a great basis for a song? By focusing on a Biblical passage or verse, you will improve the Christian content of the song and strengthen its value. This will keep it grounded in truth.\n You can listen to the sermon your preacher gives Sundays, and write about the topic he or she put forward. Try putting lines he or she emphasizes as the most important lines in your song. He or she may even be willing to help your song by reading what you've produced and make some suggestions.\n Will it tell a story? Will it illustrate a journey, or demonstrate a viewpoint?\n\n, This will help you to connect best with the audience, by writing about how you are feeling. If you're sad write a slow, steady song. If you're happy write a more upbeat song. Every good songwriter has a mixture. And the audience will connect with that, relating it back to their own experiences.\n It should be something that works for you, that you enjoy listening to and that you want to share with others. Any kind of music can be turned into a Christian song, with imagination and will.\n They may be Christian-based or secular; it doesn't really matter as you'll be using them for inspiration, not copying exactly.\n Having a general order in which to do things can get your brain working.\n Can you create a bunch of mini-songs which can then be strung together to form the whole content of the song?\n\n, If the verses added prevent the song from making good sense, move them around a bit. Shift in new ones and pull other ones out. Go back constantly to your theme or message decided earlier, to make sure the song is fitting well.\n Sing it or play it out loud, at its most basic. Is it coming across in the way you want it to? This process can be lengthy, so don't rush it. Give it plenty of time and focus on what you want the audience to understand from your song and its message.\n", 'Your teacher has worked hard to create rules for his or her classroom, and nothing is more annoying than a student who undermines them. Acting completely oblivious of the rules is just downright annoying. Here are some ways to really get under your teacher\'s skin:\n\n\nWaste class time by asking your teacher things that he clearly told you several times. This works best if you ask him something that is clearly written on the board or on the syllabus.\nIf you missed class, instead of looking at the syllabus or asking a classmate, ask your teacher, "What did I miss?"\n\nIf you really want to annoy your teacher, say, "Did I miss anything?"\n\n\nWhen your teacher reprimands you for doing something you were instructed not to do, act really oblivious. Say, "I\'m sorry, my last teacher said that was okay!"\n\n, Absolutely nothing is more annoying than a student who not only doesn\'t pay attention, but also creates a disturbance and distracts the teacher as well as other students. Here are some ways to create a racket:\n\n\nShow up late--loudly. Run into the classroom loudly and pant, "Sorry I\'m late!" Huff and puff and drop your things everywhere, creating a noisy mess. When you finally take your seat, repeat your favorite phrase, "Did I miss anything?"\nTalk to other students. Actively talk to other students while the teacher is talking, even if they don\'t want to talk to you. This is especially annoying if you are asking them a question that you should just ask the teacher.\n\nIf you want to reach astronomic levels of annoying-ness, ask your teacher a question, and then talk to your classmates while he or she tries to earnestly answer you. That will go over really well!\n\n\nUse your cell phone throughout class. Let it sit on your desk and vibrate repeatedly. Don\'t bother putting it on silent. Or you can have an incredibly loud and obnoxious ring tone. Let it ring while your phone is buried deep in your bag so it\'ll take forever to turn it off. This will surely disrupt the class and will make your classmates crack up! Your teacher will love it.\n All teachers love a student who thinks he knows more than they do. This works particularly well if your teacher is a true expert on a subject and you know absolutely nothing about it. Here are some ways to be a true know-it-all:\n\n\nAfter everything your teacher says, say, "How can you be so sure?" If your teacher plays along and tries to explain to you why something is true, grunt or say, "I guess that makes sense," but look obviously unconvinced.\nIf your teacher reprimands you, roll your eyes and sigh. That will work wonders!\nConstantly reference your other teachers, your parents, or even your friends as true expert. After everything your teacher says, say, "But my dad says..."\nLet your teacher know if you think you deserved a higher grade on a mediocre assignment. That will earn you lost of brownie points.', "The slow soak method is a preferred method of preparing beans if you've set aside enough time to soak overnight. Slow soaking ensures that the final product is fully cooked, not crunchy or underdone.\n\n\nIf you choose the slow soak method, put the beans in a 5 qt. (4.7 l) saucepan and cover them with about 8 cups (about 2 l) of water. Put the lid on the saucepan and allow them to soak overnight in the refrigerator.\n For a faster soak, put the beans and water in a saucepan and bring them to a boil. Allow them to boil for about 2 to 3 minutes. Remove the saucepan from the heat, cover it with a lid and let the beans soak for at least 1 hour.\n If you leave your beans in cool water overnight, expect them to at least double in size. Make sure your cooking vessel is large enough to accommodate this transformation., Your beans are now ready to be cooked.", 'Your contract is your legal agreement which you must abide by. If you fail to uphold your end of the bargain, then the other party can sue you in court and get money damages.Accordingly, you must follow your obligations in the lease agreement.\n\n\nBefore you can do anything, you should find your copy of the lease. Make sure it is the most recent. If you have had a long-term relationship with the tenant or facility, then you might have multiple contracts.;\n, These provisions will explain how either party can end the lease before the expiration date and the circumstances in which they can terminate the lease. In a long-term contract, there are probably several situations in which either side can end the relationship:The sporting facility was destroyed by natural disasters. If a fire or tornado destroyed the facility, then the tenant is usually given the right to terminate the contract.\nThe facility is condemned by the government. In this situation, the tenant can terminate the contract.\nOne side pays “liquidated damages.” These damages are also called an “early termination fee.” You can usually end a long-term contract early by paying a lump-sum of money to the other side.\nEither side “defaults” on its obligations. Both landlords and tenants have obligations, which are called “covenants” or “agreements” in the lease. A tenant, for example, agrees to pay rent on a regular schedule. If either party fails to uphold its obligations, then the other party can usually terminate after a certain amount of time.\n Not all sporting facility contracts are long-term. For example, you might have rented out an auditorium in order to host a two-day expo. However, if your plans change, then you might need to cancel the event. A short-term contract should have a cancellation provision.\n\n\nCancellation provisions can be pretty strict. For example, the contract might state that you have no right to a refund if you don’t cancel early enough., You should meet with an attorney to discuss your options for ending the contract before the expiration date. Your goal should be to end the contract without inviting a lawsuit. Your lawyer can help you find ways to do that.\n\n\nIf you own a sports team or a facility, then you should have a lawyer as general counsel. Call him or her and schedule a meeting. The lawyer can read the contract and come up with a strategy for making a clean break from the contract.\nIf you don’t have a lawyer, then get referrals from other teams or facility owners in your area. They might have tried to end a sporting facility contract recently and can give you the name of their lawyer.Call the lawyer up and explain your problem. You can then schedule a longer consultation.\n You might not want to pay liquidated damages to end the contract early. Instead, your ideal resolution might be to end the contract without having to pay any money. In this situation, you will have to find a promise the other side has not fulfilled, which is called a “breach.” Go through the contract cover to cover and identify every promise the other side has made.\n\n\nFor example, the stadium owner probably promises to fix hazards and not to interfere with the tenant’s enjoyment of the property. Failure to fulfill these obligations qualifies as a “breach” of the contract, which can justify termination.\nThe tenant also makes many promises, such as promising to make rent on schedule and to use the facility for only authorized purposes. If the tenant fails to fulfill these obligations, then it has also breached the contract., Before you can end the contract, you need solid evidence that you have a reason for doing so. This means getting evidence that the other side isn’t fulfilling its obligations under the lease agreement. Gather relevant evidence:\n\n\nHold onto communications with the other side. For example, you might have sent notices that rent is due. Keep copies of these notices, as well as anything the tenant sent in return.\nTake photographs. The landlord might have refused to fix a hazard. You should take photographs of the hazard to serve as proof.\nSave newspaper clippings. The tenant might have used the facility for an unauthorized purpose. Hold onto any newspaper clippings which discuss the event.\n You need to put the other side on notice of how it is not fulfilling its contractual obligations. You can do this by sending a breach of contract notice. Format the notice like a standard business letter.\n\n\nBe sure to include the date near the top of the letter. The date is important.If your contract states that the other side has 30 days to fix a problem, then the date on your letter starts the clock.\n Quote provisions from the lease agreement that the other side has not fulfilled. For example, if the tenant has not paid rent, then quote the rent payment provision.\n\n\nList every breach you can find. Start with the most serious violation and then move down to less serious ones., Here is where things can be difficult. You may want to get out of the contract immediately. However, the lease probably states that the party in breach has a chance to “cure” or fix the violation.Your contract might not let you terminate the lease until 30 days have passed and the other side has made no effort to cure.You should talk with your lawyer about how you want to handle this. If you agreed to give the other side a chance to cure, you are legally bound to that agreement. The other side might quickly fix the problem so that the contract can continue in force.\nA good way to think about the breach of contract notice is that it is an opening move in the negotiation process. You can state, “Please contact me to discuss how to resolve these issues.”\n\n, Your lease probably states where you should send the notice. Make sure to send it to the correct address and use the method identified in the lease for delivery.If you fail to use the agreed delivery method, then the other side could claim that it hasn’t received proper notice and therefore doesn’t need to fix the problems.\n\n\nIf you mail the letter, then send it certified mail, return receipt requested.\nAlways hold onto a copy of all communications with the other side.\n Negotiating a settlement would allow you to avoid going to trial, which will be more expensive than negotiations.A settlement also provides closure. You could end the contract and then move on. You should only negotiate if you are willing to compromise to reach a resolution.\n\n\nYou shouldn’t negotiate if you are unwilling to compromise. Negotiations will only be successful if you approach them with an open mind and in good faith.\nIf you really don’t want to negotiate, then you should either pay liquidated damages to end the contract early or wait to be sued and defend yourself at trial.\n You need a goal before going into negotiations. For example, if you are the tenant, then your goal might be to get out of the lease and not have to pay anything. This can be your ideal resolution.\n\n\nTo figure out how aggressive you can be during negotiations, you should consider your best alternative to a negotiated agreement. Depending on your evidence, for example, your best alternative could be to pay liquidated damages (if your evidence is weak) or win a lawsuit (if your evidence is strong). The attractiveness of your best alternative can help you decide how aggressive to be in negotiations.\nYou also need to know the absolute minimum you are willing to settle for. For example, you might be willing to settle by paying 33% of what remains on your sports facility contract. This is called your “walkaway” point.If the other side can’t meet it, then you stop negotiating.\n Negotiations will probably take place at a lawyer’s office. You should let your attorney handle most of the discussions, but be sure to offer your input. Your lawyer can’t accept a settlement offer without your approval, so stay engaged.Don’t expect to finish negotiations all in one meeting. You should continue to meet as long as you feel that you are making progress.\n If you reach an agreement, your lawyers will draft a settlement agreement for each party to sign. The settlement agreement is a new contract between the two parties. If either of you violate it, then you could be sued.\n\n\nFor example, the tenant might agree to pay some money to the landlord. If the tenant refuses, the landlord can go into court with the settlement agreement and sue.\nIf both parties agree to end the contract, then you will sign a “mutual rescission agreement.” This agreement will formally end your relationship., If negotiations fail, then you can simply stop performing under the contract. For example, a tenant could move out and stop paying rent. You would then wait for the landlord to sue you for breach of contract. Your lawyer will help you build your defense for the lawsuit.\n\n\nLawsuits can be drawn-out and time-consuming. In particular, lawsuits involve a fact-finding phase called “discovery.” In discovery, you and the other side can request documents from each other and ask each other questions under oath.The discovery phase can often last up to a year.\nAt the trial, each side will present evidence that justifies its actions. If you are trying to end the contract, then you will try to prove that the other side breached the contract first. In effect, you will try to prove that what you stated in your breach of contract notice is true.\nIf you lose at trial, you generally have the ability to bring an appeal.']


if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
# generate_kwargs = dict(max_new_tokens=num_tokens, use_cache=False, do_sample=False)
# generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=False)

print_rank0(f"Generate args {generate_kwargs}")
inputs = input_sentences[: args.batch_size]


def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(inputs, max_length=350,
            padding="max_length",
            truncation=True,
            return_tensors="pt",)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    outputs = model.generate(**input_tokens, num_beams=5,max_new_tokens=150,min_length=5)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


print_rank0("*** Running generate")
t_generate_start = time.time()
generated = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in generated:
    print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")


### Benchmark

if args.benchmark:
    # clear cache / free memory
    torch.cuda.empty_cache()
    gc.collect()

    print_rank0("*** Running benchmark")
    # warm up
    for i in range(1):
        _ = generate()
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate()
        total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)
    torch.cuda.synchronize()
    througput = (time.time() - t0) / (total_new_tokens_generated)
    print_rank0(
        f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
"""
    )