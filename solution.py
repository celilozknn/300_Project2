from mpi4py import MPI
import argparse
import string
from collections import Counter


def preprocess_sentence(sentence, stopwords):
    """
    Apply lower-casing, punctuation removal and stopword removal
    :param sentence: input sentence to be preprocessed
    :param stopwords: commonly used words such as “the”, “is”, “to”, and “in”,
    which are called stopwords
    :return: list of valid words
    """
    # Lower-case
    sentence = sentence.lower()
    # Remove punctuation
    clean_sentence = ""
    for char in sentence:
        if char not in string.punctuation:
            clean_sentence += char
    # Remove stopwords
    words = clean_sentence.split()
    clean_words = [w for w in words if w not in stopwords]

    return clean_words


def compute_tf(sentences_data, vocab_set):
    """
    Counts number of each word in the vocab_set across all sentences combined:
    :param sentences_data: list of preprocessed words lists
    :param vocab_set: list of vocab words
    :return: tf_count
    """
    tf_counts = Counter()
    for words in sentences_data:
        for w in words:
            if w in vocab_set:
                tf_counts[w] += 1
    return tf_counts


def compute_df(sentences_data, vocab_set):
    """
    Counts number of distinct sentences in which each word in the vocabulary appears
    :param sentences_data: list of preprocessed words lists
    :param vocab_set: list of vocab words
    :return: df_count
    """
    df_counts = Counter()
    for words in sentences_data:
        # Use set to count a word only once per sentence
        unique_words = set(words)
        for w in unique_words:
            if w in vocab_set:
                df_counts[w] += 1
    return df_counts


def split_into_chunks(data, n_chunks):
    """
    Splits all data into n chunks
    :param data: input data to be divided
    :param n_chunks: number of chunks
    :return: list of chunks with n_chunks elements
    """
    total_len = len(data)
    k = total_len // n_chunks  # k is the base size (e.g., 21 // 5 = 4)
    m = total_len % n_chunks  # m is the remainder (e.g., 21 % 5 = 1)

    chunks = []
    current_index = 0

    for i in range(n_chunks):
        if i < m:
            chunk_size = k + 1
        else:
            chunk_size = k
        start = current_index
        end = current_index + chunk_size
        chunks.append(data[start:end])
        current_index = end  # update current index for next chunk
    return chunks


def pattern1(comm, rank, size, text_lines, vocab_set, stopwords_set):
    """
    Pattern #1: Parallel End-to-End Processing
    Manager distributes balanced chunks. Workers do all NLP + TF.
    :param comm: default communicator that contains all processes launched by mpiexec
    :param rank: rank of the processor
    :param size: size of the processor
    :param text_lines: input text to be processed
    :param vocab_set: vocabulary set
    :param stopwords_set: commonly used words such as “the”, “is”, “to”, and “in”, which are called stopwords
    :return: tf_count (if manager aka rank=0)
    """
    if rank == 0:  # Manager
        num_workers = size - 1
        chunks = split_into_chunks(text_lines, num_workers)
        # Distribute
        for i in range(num_workers):
            dest = i + 1
            # Send config first (manual broadcast)
            comm.send((vocab_set, stopwords_set), dest=dest)
            # Send data
            comm.send(chunks[i], dest=dest)
        # Aggregate
        total_tf = Counter()
        for i in range(num_workers):
            partial_tf = comm.recv(source=i + 1)
            total_tf.update(partial_tf)

        return total_tf, None  # DF is None for Pattern 1

    else:
        # Worker
        vocab, stopwords = comm.recv(source=0)
        lines = comm.recv(source=0)

        processed_data = []
        for line in lines:
            processed_data.append(preprocess_sentence(line, stopwords))

        tf_counts = compute_tf(processed_data, vocab)
        comm.send(tf_counts, dest=0)
        return None, None


def pattern2(comm, rank, size, text_lines, vocab_set, stopwords_set):
    """
    Pattern #2: Linear Pipeline (5 Processes Fixed)
    M -> W1(Lower) -> W2(Punct) -> W3(Stop) -> W4(TF) -> M
    :param comm: default communicator that contains all processes launched by mpiexec
    :param rank: rank of the processor
    :param size: size of the processor
    :param text_lines: input text to be processed
    :param vocab_set: vocabulary set
    :param stopwords_set: commonly used words such as “the”, “is”, “to”, and “in”, which are called stopwords
    :return: tf_count (if manager aka rank=0)
    """
    if rank == 0:   # manager process
        # Send config to workers
        for r in range(1, 5):
            comm.send((vocab_set, stopwords_set), dest=r)
        # Split chunks and send it to workers
        num_chunks = 10
        chunks = split_into_chunks(text_lines,num_chunks)
        for chunk in chunks:
            comm.send(chunk, dest=1)
        # Send termination signal
        comm.send(None, dest=1)
        final_tf = comm.recv(source=4)
        return final_tf, None

    else:   # worker process
        # Receive Config
        vocab, stopwords = comm.recv(source=0)
        # Initialize accumulator only for the last worker
        if rank == 4:
            acc_tf = Counter()
        else:
            acc_tf = None

        while True:
            # Receive from previous stage
            source = rank - 1
            data = comm.recv(source=source)
            # Termination check
            if data is None:
                if rank < 4:
                    # Pass the "None" signal to the next worker
                    comm.send(None, dest=rank + 1)
                else:
                    # Rank 4 is done, send result to Manager
                    comm.send(acc_tf, dest=0)
                break
            # Processing logic
            processed_chunk = []
            if rank == 1:  # Lower-casing
                for line in data:
                    processed_chunk.append(line.lower())
                comm.send(processed_chunk, dest=2)
            elif rank == 2:  # Punctuation Removal
                # Manual loop implementation
                for line in data:
                    clean_line = ""
                    for char in line:
                        if char not in string.punctuation:
                            clean_line += char
                    processed_chunk.append(clean_line)
                comm.send(processed_chunk, dest=3)
            elif rank == 3:  # Stopword Removal
                for line in data:
                    words = line.split()
                    clean = [w for w in words if w not in stopwords]
                    processed_chunk.append(clean)
                comm.send(processed_chunk, dest=4)
            elif rank == 4:  # Term Frequency (TF)
                partial_tf = compute_tf(data, vocab)
                acc_tf.update(partial_tf)
        return None, None


def pattern3():
    pass

def pattern4():
    pass


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = None
    text_lines = []
    vocab_set = set()
    stopwords_set = set()
    pattern = 0

    # Manager reads files
    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", required=True)
        parser.add_argument("--vocab", required=True)
        parser.add_argument("--stopwords", required=True)
        parser.add_argument("--pattern", type=int, required=True)
        args = parser.parse_args()

        # Read text lines
        text_lines = []
        file_text = open(args.text, 'r', encoding='utf-8')
        for line in file_text:
            line = line.strip()
            if line:  # Check if line is not empty
                text_lines.append(line)
        file_text.close()

        # Read vocabulary
        vocab_set = set()
        file_vocab = open(args.vocab, 'r', encoding='utf-8')
        for line in file_vocab:
            word = line.strip()
            if word:
                vocab_set.add(word)
        file_vocab.close()

        # Read stopwords
        stopwords_set = set()
        file_stop = open(args.stopwords, 'r', encoding='utf-8')
        for line in file_stop:
            word = line.strip()
            if word:
                stopwords_set.add(word)
        file_stop.close()
        pattern = args.pattern
    # To strictly follow "No Collectives", we send pattern ID to everyone first.
    if rank == 0:
        for i in range(1, size):
            comm.send(pattern, dest=i)
    else:
        pattern = comm.recv(source=0)

    # Execute Selected Pattern
    final_tf, final_df = None, None

    if pattern == 1:
        final_tf, final_df = pattern1(comm, rank, size, text_lines, vocab_set, stopwords_set)
    elif pattern == 2:
        final_tf, final_df = pattern2(comm, rank, size, text_lines, vocab_set, stopwords_set)
    elif pattern == 3:
        pass
    elif pattern == 4:
        pass

    # Print Results
    if rank == 0:
        print("Term-Frequency (TF) Result:")
        sorted_vocab = sorted(list(vocab_set))
        for w in sorted_vocab:
            print(f"{w}: {final_tf.get(w, 0)}")
        if final_df is not None:
            print("\nDocument-Frequency (DF) Result:")
            for w in sorted_vocab:
                print(f"{w}: {final_df.get(w, 0)}")
