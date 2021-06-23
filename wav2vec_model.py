from lib2to3.pgen2.tokenize import tokenize
import numpy as np 
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Model,Wav2Vec2Tokenizer
from transformers import AdamW
# from torch.utils.data import DataLoader,Dataset
import os
import edit_distance
from transformers.file_utils import PaddingStrategy
import time

class audio_dataset():
    def __init__(self, split = 'train'):
        self.split = split
        self.prefix = '/data1/private/houbairu/audio_dataset/orig_librispeech/' + self.split + "/"
        self.filename_list, self.text_list = self.read_file()


    def read_file(self):
        filename_list = []
        text_list = []
        dir_list = os.listdir(self.prefix)
        # print(dir_list)
        for dir_name in dir_list:
            total_dir_name = self.prefix + dir_name + "/"
            sub_dir_list = os.listdir(total_dir_name)
            for sub_dir_name in sub_dir_list:
                text_file = dir_name + '-' + sub_dir_name + '.trans.txt'
                with open(self.prefix + dir_name + "/" + sub_dir_name + "/" + text_file,'r', encoding = 'utf-8') as f:
                    for line in f:
                        items = line.strip().split()
                        audio_file_name = items[0]
                        trans_texts = items[1:]
                        filename_list.append(self.prefix + dir_name + "/" + sub_dir_name + "/" + audio_file_name + '.flac')
                        text_list.append(' '.join(trans_texts))
        return filename_list, text_list
 



class wav2vec_model_finetuned():
    def __init__(self, model_type = 'facebook/wav2vec2-base-100h', cache_dir = '/data1/private/houbairu/model_cache/hg_wav2vec2/',):
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        # self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        self.text_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)

        self.audio_tokenizer.do_normalize = True
        self.audio_tokenizer.return_attention_mask = True

        self.audio_max_length = 20 * 16000
        self.model = self.model.to("cuda")
        # self.model.half()

    def tokenize_text(self, text_list):
        # res = self.text_tokenizer(text_list, padding = "longest", return_tensor = 'pt', return_attention_mask = True)
        res = self.text_tokenizer(text_list, padding = "longest", return_attention_mask = True)
        return res

    def tokenize_audio(self, audio_list):
        # res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = True, max_length = 30*16000)
        # res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = PaddingStrategy.MAX_LENGTH, max_length = 20*16000, truncation = True)
        res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = "longest")

        return res

    def predict(self,audio_list, text_list):
        tokenized_audio = self.tokenize_audio(audio_list)
        audio_inputs = tokenized_audio.input_values
        audio_attention_mask = tokenized_audio.attention_mask

        if audio_inputs.size(1) > self.audio_max_length:
            audio_inputs = audio_inputs[:, :self.audio_max_length]
            audio_attention_mask = audio_attention_mask[:, :self.audio_max_length]
        audio_inputs = audio_inputs.to("cuda")
        audio_attention_mask = audio_attention_mask.to("cuda")


        result = self.model(audio_inputs, attention_mask = audio_attention_mask)
        logits = result.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.text_tokenizer.batch_decode(predicted_ids)
        return transcription

    def test_model(self):
        time.sleep(30)
        self.model.eval()
        batch_size = 2
        dataset = audio_dataset("test")
        text_list = dataset.text_list
        audio_file_list = dataset.filename_list
        batch_num = len(text_list) // batch_size
        pred_text_list = []
        for i in range(batch_num):
            print(i,'/', batch_num)
            audio_list = []
            for filename in audio_file_list[i * batch_size: (i + 1) * batch_size]:
                audio_array = sf.read(filename)[0]
                audio_list.append(audio_array)
            res = self.predict(audio_list, text_list)
            pred_text_list += res
        if batch_num * batch_size < len(text_list):
            # print(i,'/', batch_num)
            audio_list = []
            for filename in audio_file_list[batch_num * batch_size:]:
                audio_array = sf.read(filename)[0]
                audio_list.append(audio_array)
            res = self.predict(audio_list, text_list)
            pred_text_list += res            
        
        print(len(text_list), len(pred_text_list))
        assert len(text_list) == len(pred_text_list)

        err = 0
        total = 0
        for i in range(len(text_list)):
            word_list = text_list[i].split()
            pred_list = pred_text_list[i].split()
            distance = edit_distance.edit_distance(word_list, pred_list)
            err += distance[0]
            total += len(word_list)
        print("WER: ", err)
        print("total: ", total)
        print("error: ", err/total)
    
class wav2vec_model():
    def __init__(self, model_type = 'facebook/wav2vec2-base', cache_dir = '/data1/private/houbairu/model_cache/hg_wav2vec2/',):
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir).to("cuda")
        # self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        self.text_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)

        self.audio_tokenizer.do_normalize = True
        self.audio_tokenizer.return_attention_mask = True

    def tokenize_text(self, text_list):
        # res = self.text_tokenizer(text_list, padding = "longest", return_tensor = 'pt', return_attention_mask = True)
        res = self.text_tokenizer(text_list, padding = "longest", return_attention_mask = True)
        return res

    def tokenize_audio(self, audio_list):
        # res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = True, max_length = 30*16000)
        # res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = True, max_length = 30*16000)
        res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = "longest", max_length = 30*16000)

        return res

    def predict(self,audio_list, text_list):
        tokenized_audio = self.tokenize_audio(audio_list)
        tokenized_text = self.tokenize_text(text_list)

        text_input_ids = tokenized_text.input_ids
        text_attention_mask = tokenized_text.attention_mask

        audio_inputs = tokenized_audio.input_values
        audio_attention_mask = tokenized_audio.attention_mask

        result = self.model(audio_inputs, attention_mask = audio_attention_mask)
        logits = result.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.text_tokenizer.batch_decode(predicted_ids)
        return transcription

    def finetune(self, batch_size = 8, epoch = 20):
        optimizer = AdamW(self.model.parameters(), lr = 1e-4)
        train_set = audio_dataset("train")
        valid_set = audio_dataset("dev")
        test_set = audio_dataset("test")
        train_text_list, train_audio_file_list = train_set.text_list, train_set.filename_list
        valid_text_list, valid_audio_file_list = valid_set.text_list, valid_set.filename_list
        test_text_list, test_audio_file_list = test_set.text_list, test_set.filename_list

        batches_per_epoch = len(train_text_list)//batch_size
        for epoch_idx in range(epoch):
            self.model.train()
            selection = np.random.choice(len(train_text_list),size = len(train_text_list),replace = False)
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for batch_idx in range(batches_per_epoch):
                select_idx_list = selection[idx * batch_size, (idx+1) * batch_size]
                
                audio_list = []
                text_list = []
                for idx in select_idx_list:
                    filename = train_audio_file_list[idx]
                    audio_array = sf.read(filename)[0]
                    audio_list.append(audio_array)
                    text_list.append(train_text_list[idx])

                tokenized_audio = self.tokenize_audio(audio_list)
                tokenized_text = self.tokenize_text(text_list)

                text_input_ids = tokenized_text.input_ids

                audio_inputs = tokenized_audio.input_values
                audio_attention_mask = tokenized_audio.attention_mask

                res = self.model(audio_inputs, labels = text_input_ids)
                loss = res.loss

                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            epoch_loss /= batches_per_epoch
            print("epoch loss: ", epoch_idx, "  ", epoch_loss)

            train_acc = self.evaluate_accuarcy(train_text_list, train_audio_file_list)
            print("train WER: ", train_acc)

            valid_acc = self.evaluate_accuarcy(valid_text_list, valid_audio_file_list)
            print("valid WER: ", valid_acc)


        test_acc = self.evaluate_accuarcy(test_text_list, test_audio_file_list)
        print("test WER: ", test_acc)     




    def evaluate_accuarcy(self, test_text_list, test_audio_file_list):
        self.model.eval()
        batch_size = 3
        dataset = audio_dataset("test")
        text_list = test_text_list
        audio_file_list = test_audio_file_list
        batch_num = len(text_list) // batch_size
        pred_text_list = []
        for i in range(batch_num):
            print(i,'/', batch_num)
            audio_list = []
            for filename in audio_file_list[i * batch_size: (i + 1) * batch_size]:
                audio_array = sf.read(filename)[0]
                audio_list.append(audio_array)
            res = self.predict(audio_list, text_list)
            pred_text_list += res
        
        assert len(text_list) == len(pred_text_list)

        err = 0
        total = 0
        for i in range(len(text_list)):
            word_list = text_list[i].split()
            pred_list = pred_text_list[i].split()
            distance = edit_distance.edit_distance(word_list, pred_list)
            err += distance[0]
            total += len(word_list)
        print("WER: ", err)
        print("total: ", total)
        print("error: ", err/total)
    

class wav2vec_model_parallel():
    def __init__(self, model_type = 'facebook/wav2vec2-base', cache_dir = '/data1/private/houbairu/model_cache/hg_wav2vec2/facebook_wav2vec2-base/',
                    gpu_num = 4):
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir).to("cuda")
        # self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        self.text_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)
        self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained(pretrained_model_name_or_path = model_type, cache_dir = cache_dir)

        self.audio_tokenizer.do_normalize = True
        self.audio_tokenizer.return_attention_mask = True
        device_list = [i for i in range(gpu_num)]
        self.model = torch.nn.DataParallel(self.model, device_ids = device_list)
        self.optimizer = AdamW(self.model.module.parameters(), lr = 1e-4)


    def tokenize_text(self, text_list):
        # res = self.text_tokenizer(text_list, padding = "longest", return_tensor = 'pt', return_attention_mask = True)
        res = self.text_tokenizer(text_list, padding = "longest", return_attention_mask = True)
        return res

    def tokenize_audio(self, audio_list):
        # res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = True, max_length = 30*16000)
        # res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = True, max_length = 30*16000)
        res = self.audio_tokenizer(audio_list, return_tensors = 'pt', padding = "longest", max_length = 30*16000)

        return res

    def predict(self,audio_list, text_list):
        tokenized_audio = self.tokenize_audio(audio_list)
        tokenized_text = self.tokenize_text(text_list)

        # text_input_ids = tokenized_text.input_ids.to("cuda")
        # text_attention_mask = tokenized_text.attention_mask

        audio_inputs = tokenized_audio.input_values.to("cuda")
        audio_attention_mask = tokenized_audio.attention_mask.to("cuda")

        result = self.model(audio_inputs, attention_mask = audio_attention_mask)
        logits = result.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.text_tokenizer.batch_decode(predicted_ids)
        return transcription

    def finetune(self, batch_size = 8, epoch = 20):

        output_dir = "./finetuned/"

        train_set = audio_dataset("train")
        valid_set = audio_dataset("dev")
        test_set = audio_dataset("test")
        train_text_list, train_audio_file_list = train_set.text_list, train_set.filename_list
        valid_text_list, valid_audio_file_list = valid_set.text_list, valid_set.filename_list
        test_text_list, test_audio_file_list = test_set.text_list, test_set.filename_list

        batches_per_epoch = len(train_text_list)//batch_size

        global_WER = 10000
        for epoch_idx in range(epoch):
            self.model.train()
            selection = np.random.choice(len(train_text_list),size = len(train_text_list),replace = False)
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for batch_idx in range(batches_per_epoch):
                select_idx_list = selection[idx * batch_size, (idx+1) * batch_size]
                
                audio_list = []
                text_list = []
                for idx in select_idx_list:
                    filename = train_audio_file_list[idx]
                    audio_array = sf.read(filename)[0]
                    audio_list.append(audio_array)
                    text_list.append(train_text_list[idx])

                tokenized_audio = self.tokenize_audio(audio_list)
                tokenized_text = self.tokenize_text(text_list)

                text_input_ids = torch.LongTensor(tokenized_text.input_ids).to("cuda")

                audio_inputs = tokenized_audio.input_values.to("cuda")
                audio_attention_mask = tokenized_audio.attention_mask.to("cuda")

                loss, _ = self.model(audio_inputs, labels = text_input_ids, attention_mask = audio_attention_mask,return_dict = False)
                avg_loss = torch.mean(loss)

                self.optimizer.zero_grad()
                epoch_loss += avg_loss.item()
                avg_loss.backward()
                self.optimizer.step()
            
            epoch_loss /= batches_per_epoch
            print("epoch loss: ", epoch_idx, "  ", epoch_loss)

            train_WER = self.evaluate_accuarcy(train_text_list, train_audio_file_list)
            print("train WER: ", train_WER)

            valid_WER = self.evaluate_accuarcy(valid_text_list, valid_audio_file_list)
            print("valid WER: ", valid_WER)

            if valid_WER < global_WER:
                global_WER = valid_WER
                self.model.module.save(output_dir + "best.pt")
        test_acc = self.evaluate_accuarcy(test_text_list, test_audio_file_list)
        print("test WER: ", test_acc)     




    def evaluate_accuarcy(self, test_text_list, test_audio_file_list):
        self.model.eval()
        batch_size = 3
        dataset = audio_dataset("test")
        text_list = test_text_list
        audio_file_list = test_audio_file_list
        batch_num = len(text_list) // batch_size
        pred_text_list = []
        for i in range(batch_num):
            print(i,'/', batch_num)
            audio_list = []
            for filename in audio_file_list[i * batch_size: (i + 1) * batch_size]:
                audio_array = sf.read(filename)[0]
                audio_list.append(audio_array)
            res = self.predict(audio_list, text_list)
            pred_text_list += res
        
        assert len(text_list) == len(pred_text_list)

        err = 0
        total = 0
        for i in range(len(text_list)):
            word_list = text_list[i].split()
            pred_list = pred_text_list[i].split()
            distance = edit_distance.edit_distance(word_list, pred_list)
            err += distance[0]
            total += len(word_list)
        print("WER: ", err)
        print("total: ", total)
        print("error: ", err/total)
    


if __name__== "__main__":

    model = wav2vec_model_finetuned()
    model.test_model()



