# Neural-Language-Modeling

## 1. Running the Language Model Generator

The `generator.py` script allows users to generate the most probable next word given an input sentence using different language models.

### 1.1 Usage

python3 generator.py <lm_type> <corpus_path> 

### 1.2 Arguments
- **`lm_type`**: Specifies the type of language model to use. Available options:
  - `-f` : Feedforward Neural Network (FFNN)
  - `-r` : Recurrent Neural Network (RNN)
  - `-l` : Long Short-Term Memory (LSTM)
- **`corpus_path`**: Path to the corpus file.
- **`k`**: Number of top candidate words to display.

---

## 2. Example Usage

### 2.1 Running the FFNN Model
```bash
python3 generator.py -f ./corpora/external/{corpus}.txt 3
```
### 2.2 Running the RNN Model
```bash
python3 generator.py -r ./corpora/external/{corpus}.txt 3
```
### 2.3 Running the LSTM Model

```bash
python3 generator.py -l ./corpora/external/{corpus}.txt 3
```

where corpora can be ```Pride and Prejudice - Jane Austen``` or ```Ulysses - James Joyce```

---

## 3. Output Format
- The script prompts the user for an input sentence.
- It predicts the most probable next words along with their probability scores.

### 3.1 Example Execution

```bash
python3 generator.py -f ./corpora/external/{corpus}.txt 3
```
**User Input:**
```
Input sentence: An apple a day keeps the doctor
```
**Output:**

```
away 0.4
happy 0.2
fresh 0.1
```
---

## 4. Pretrained models:

When you execute the above code, you will be using pretrained models.

If you wish to train your own models:  
1. Import the notebook ```Training.ipynb``` to kaggle.
2. Turn on the ```P100``` GPU 
3. Click on Run all
4. Then, after a while, you will see an ```everything.zip``` file in ```/kaggle/working``` directory, download it (it may take some time).
5. Change the names of the files: lnn -> lstm, fnn -> ffnn. (Sorry for not implementing this)

## 5. Notes
- Ensure all dependencies are installed before running the script.
- The script processes the provided corpus and trains the model before generating predictions.
- You can include additional modular files for better organization.