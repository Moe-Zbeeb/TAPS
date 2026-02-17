import argparse


parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--outdir', type=str, default='outdir0')
parser.add_argument('--data_path', type=str, default='0')
parser.add_argument('--model_path', type=str, default='0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


bigname = args.model_path



def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files=args.data_path)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles=["user","assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                if sentence["from"]=="gpt":
                    sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            conversation=tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)
            #print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"



            total_len = len(input_ids)

            sep2="<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1]=turns[0]+sep2+turns[1]
            turns=turns[1:]


            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1


                # Ignore the user instructions
                if i==0:
                    loss_mask[cur_len: cur_len + instruction_len-2] = 0
                else:
                    loss_mask[cur_len-3: cur_len + instruction_len+1] = 0
                cur_len += turn_len
                if i!=0:
                    cur_len+=3
                #cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0



            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        #num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1

bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
if bigtokenizer.pad_token_id is None:
    # Fall back to eos/unk for padding so batching works
    bigtokenizer.pad_token_id = bigtokenizer.eos_token_id or bigtokenizer.unk_token_id
ds = build_dataset_rank(bigtokenizer)
print(ds)
# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
#bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)
bigmodel.eval()
device = next(bigmodel.parameters()).device



@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    lengths = data["lengths"]
    outs_big = bigmodel(input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1].cpu()
    loss_masks = data["loss_mask"].cpu()
    outputs = []
    for i, length in enumerate(lengths):
        outputs.append(
            {
                "input_ids": input_ids.cpu()[i, :length],
                "hidden_state": hidden_state_big[i, :length],
                "loss_mask": loss_masks[i, :length],
            }
        )
    return outputs

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

def collate_batch(batch, pad_token_id):
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    loss_mask = [item["loss_mask"].squeeze(0) for item in batch]
    lengths = [len(x) for x in input_ids]
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_loss = pad_sequence(loss_mask, batch_first=True, padding_value=0)
    attention_mask = (padded_inputs != pad_token_id).long()
    return {
        "input_ids": padded_inputs,
        "loss_mask": padded_loss,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }

dataloader = DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_batch(batch, bigtokenizer.pad_token_id),
)

written = 0
for data in dataloader:
    outdata = ge(data)
    for data_point in outdata:
        writedata(outdir, data_point)
        written += 1
        if written % 100 == 0:
            print(written, end="\t")
        if written % 1000 == 0:
            print("")
