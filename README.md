# Deep_Learning
Character‑Level Shakespeare Text Generation with a From‑Scratch Vanilla RNN
Built a pure tanh RNN from first principles in PyTorch to model character‑level Shakespeare. Trained on 1.1M characters, reducing loss from 2.97 to 1.70. Generated text shows local coherence and learned formatting; long‑range incoherence demonstrates vanishing gradients. Code modular, tested, and documented.
Generates new text that sounds Shakespearean but drifts after ~20 characters — this demonstrates the vanishing gradient problem

Deep_Learning/
vanilla-rnn-shakespeare/          ← the RNN project
        model/                        ← RNN cell and language model
        utils/                        ← data loading and vocabulary
        data/raw/shakespeare.txt      ← training data
        tests/                        ← unit tests
        checkpoints/                  ← trained model weights  
        outputs/generated_samples.txt ← the generated text

<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/8c5580ee-13cc-4605-a995-c34e3015a834" />
