import numpy as np
import logging
import spacy
import torch
import transformers
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
# to measure time
import timeit 
import time
# to load llama3
from typing import List
from transformers import pipeline
# to make output file
import os


logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", 
                    trust_remote_code = "falcon" in model_name_or_path)
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            # text가 sentence 단위인가?
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        # self.generator.generate_attn(
        # prompt, self.generate_max_length, use_entropy = self.method == "dragin", use_logprob = self.method == "attn_prob")
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt") # prompt encode
        input_ids = input_ids.to(self.model.device) 
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        
        # reproductibility
        transformers.random.set_seed(0)
        
        outputs = self.model.generate(
            input_ids = input_ids, # prompt
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
        )
        # print("outputs: ", outputs)
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])
        # print("generate_attn function) generated tokens: ", generated_tokens)
        # print("generate_attn function) new_text = generated tokens[0]: ", generated_tokens[0])
        
        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0
        # modifed for measure execution time
        self.retrieve_time = 0
        self.generate_time = 0
        self.query_time = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        # modified for measure generation time
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "retrieve_time": self.retrieve_time - other_counter.retrieve_time,
            "generate_count": self.generate - other_counter.generate,
            "generate_time": self.generate_time - other_counter.generate_time,
            "query_time": self.query_time - other_counter.query_time,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    

class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


# IRCoT
class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            # modified for measure retrieval time
            retrieve_start_time = time.perf_counter()
            docs = self.retrieve(question, topk=self.retrieve_topk)
            retrieve_end_time = time.perf_counter()
            self.counter.retrieve_time += (retrieve_end_time - retrieve_start_time)
            # generate prompt for top k
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in the same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                # modified for measure generation_time
                generation_start_time = time.perf_counter()
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                generation_end_time = time.perf_counter()
                self.counter.generate_time += (generation_end_time - generation_start_time)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
            else:
                # fix sentence
                # modified for measure generation_time
                generation_start_time = time.perf_counter()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                generation_end_time = time.perf_counter()
                self.counter.generate_time += (generation_end_time - generation_start_time)
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate( # Is 'new text' a sentence here?
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "") # Remove all [xxx] from curr
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext] # All previous sentences including the question?
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0) 
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    

class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0] # Sentence segmentation
        tid = 0
        for sid, sent in enumerate(sentences): 
            tl, tr = tid, tid 
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens) # Start and end positions of the sentence
            else: 
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq: # If the sentence is found in the segment, update tr
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns) # Normalization
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value] # Determine hallucination
            if 1 in thres: # If any token in a particular sentence is hallucinated
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent) # Convert the sentence to an object
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok): # Function to check if it is a real word
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)): # Check each token to see if it's a real word
                        if not match(tokens[tl+i]):
                            thres[i] = 0 # Cancel hallucination if it's not a real word                
                
                # If the sentence is the first sentence, use ""; otherwise, concatenate all previous sentences into one string
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, new_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        # modified for consider text after hallucination
        # question_start = new_text.find("Question:")
        # If the string "Question:" exists in the text, cut off everything after that part
        # if question_start != -1:
        #     new_text = new_text[:question_start].strip()
        all_text = prev_text + " " + curr_text + " " + new_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])
        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        tl, tr = 0, len(tokens)
        curr_st = len(tokens) - len(curr_tokens)
        attns = []
        for r in range_:
            att = torch.zeros(atten_tmp.shape[0], input_length)
            for i in range(r[0], r[1] + 1):
                att += atten_tmp[:, i]
            att /= (r[1] - r[0] + 1)
            att = torch.mean(att, dim=0)
            att = att[tl:tr]
            if att.shape[0] > 1:
                att = att / sum(att[1:]).item()
            attns.append(att)
            
        forward_attns = torch.zeros(tr - tl)
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1: # Hallucinated token
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__: # top k
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:  # A certain ratio of the total
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0])
        real_pairs = real_pairs[:top_k] # A certain number of top items
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs]) # Construct a string with tokens

    def generate_query_with_lm(self, question, prev_text, curr_tokens, curr_hit):
        # hallucinated_tokens = [token for token, hit in zip(curr_tokens, curr_hit) if hit == 1]
        hallucinated_spans = []
        span = []
        
        # If there are consecutive hallucinated tokens in curr_tokens, process them as a span
        for token, hit in zip(curr_tokens, curr_hit):
            if hit == 1:
                span.append(token)
            else:
                if span:
                    hallucinated_spans.append(" ".join(span))
                    span = []
        if span:
            hallucinated_spans.append(" ".join(span))
            
        if not hallucinated_spans:
            return ""  
        
        # hallucinated_part = ', '.join(f'"{span}"' for span in hallucinated_spans)
        # generated_text = prev_text  + " ".join(curr_tokens)  # Combine curr_tokens into a string
        curr_text = " ".join(curr_tokens)
        prompt_parts = []
        prompt_parts.append(f'{prev_text} {curr_text}\n Read the above passage and generate questions.')
        for token in hallucinated_spans:
            prompt_part = f'Generate a question to which the answer is the term/entity/phrase "{token}".'
            prompt_parts.append(prompt_part)
        prompt = "\n".join(prompt_parts)
        #print("prompt: ", prompt)
        
        # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        # inputs = tokenizer(prompt, return_tensors="pt")
        # Generate
        # generate_ids = model.generate(inputs.input_ids, max_length=100)
        # queries = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        num_return_sequences = len(hallucinated_spans) if len(hallucinated_spans) > 0 else 1

        # Meta-Llama-3-8B 
        model_id = "meta-llama/Meta-Llama-3-8B"
        llama_model = pipeline("text-generation", model=model_id)
        queries = llama_model(prompt, 
                            num_return_sequences=num_return_sequences,
                            max_length=256)
        # print("queries: ", queries)
        
        cleaned_queries = []
        for query in queries:
            generated_text = query.get("generated_text", "") 
            query_without_prompt = generated_text.split(prompt, 1)[-1].strip()
            cleaned_queries.append(query_without_prompt) 
        # Combine the queries into a single string
        combined_queries = " ".join(cleaned_queries)
        return combined_queries

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        i = 0
        
        # Define the directory and file name for the log file
        query_log_file = open("/home/shkim/DRAGIN/result/errorcase/real_words_all/103/103_query_log.txt", "a")
        while True:
            #print("step: ", i)
            
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # ('####', prompt)
            # prompt += case + " " + text
            
            # modified for measure generation_time
            query_log_file.write("============ LLM generated output ============\n")
            start_time = time.perf_counter()
            generation_code = new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                # self.attention_solver, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            end_time = time.perf_counter()
            self.counter.generate_time += (end_time - start_time)
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]
            query_log_file.write(f"LLM generate output: {new_text}\n")
            # query_log_file.write(f"tokens: {tokens}\n")
            # print("new_text) ", new_text)
            # print("tokens) ", tokens)
            # print("attns: ", attns)
            # print("logprobs: ", logprobs)
            # print("entropies: ", entropies)
            
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            query_log_file.write("============ Decide Hallucination ============\n")
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            query_log_file.write(f"hallucination 여부: {hallucination}\n")
            query_log_file.write(f"ptext: {ptext}\n")
            query_log_file.write(f"hallucinated sentence: {curr_tokens}\n")
            query_log_file.write(f"hallucinated tokens: {curr_hit}\n")
            
            if not hallucination:
                #print("not hallucination")
                query_log_file.write("============ Not Hallucination ============\n")
                text = text.strip() + " " + new_text.strip()
            else:
                query_log_file.write("============= Hallucination ============\n")
                # modified for measure generation_time (include query formulation to retrieve)
                
                forward_all = [question, text, ptext] 
                # Query, the output from previous steps in the for loop, the output generated up to the current step before hallucination
                forward_all = " ".join(s for s in forward_all if len(s) > 0) # Combine into a single string

                # Return the last num tokens of the text
                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence
                
                
                query_start_time = time.perf_counter()
                if self.query_formulation == "current": # The sentence where hallucination occurred
                    retrieve_question = " ".join(curr_tokens) 

                elif self.query_formulation == "current_wo_wrong": # Remove the tokens where hallucination occurred 
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all": # Query with all previous generated results
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence": # The last sentence from the previous generated 
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens": # The last n tokens from the previous generated 
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": # Original Dragin 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        # modified for consider tokens after hallucination
                        new_text = '',
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 

                # Query formulation method I added for this project
                # 1) Attention all context
                elif self.query_formulation == "real_words_all": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        # modified for consider tokens after hallucination
                        new_text = new_text, # after hallucination
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    )  
                # 2) LLM Generated query
                elif self.query_formulation == "generated_question": 
                    retrieve_question = self.generate_query_with_lm(
                        question = question,
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    )
                else:
                    raise NotImplemented
                query_end_time = time.perf_counter()
                self.counter.query_time += (query_end_time - query_start_time)
                
                query_log_file.write("============ Query Formulation ============\n")
                query_log_file.write(f"query_formulation_time: {query_end_time - query_start_time}\n")
                query_log_file.write(f"retrieve_query: {retrieve_question}\n")
                
                retrieve_start_time = time.perf_counter()
                
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                retrieve_end_time = time.perf_counter()
                self.counter.retrieve_time += (retrieve_end_time - retrieve_start_time)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                
                # modified for measure generation_time
                start_time = time.perf_counter()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                end_time = time.perf_counter()
                self.counter.generate_time += (end_time - start_time)
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                
                #print("after retrieve_new text) ", new_text)
                query_log_file.write("============ After Retrieval ============\n")
                query_log_file.write(f"검색 후 LLM 생성 결과: {new_text}\n")
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

                # print("### retrieve_question ###")
                # print(retrieve_question)
                # context = "### Context: ###\n"
                # for i, doc in enumerate(docs):
                #     context += f"[{i+1}] {doc}\n" 
                # print(context)
                # print(text)
                i += 1
            
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        query_log_file.write("============ Finish Inference ============\n")
        query_log_file.close()
        return text
