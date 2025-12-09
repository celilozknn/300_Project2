# Celil Özkan 2023400324
# Zeynep Ebrar Karadeniz 2022400030
# TODO: add any notes relevant for grading (e.g., known issues, partial functionality).

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

def get_pipeline_id(rank, pipeline_size=4):
    """
    Given a worker rank, returns which pipeline it belongs to.

    :param rank: global MPI rank (manager is rank 0)
    :param pipeline_size: number of workers per pipeline (default 4)
    :return: pipeline_id (0-based)
    """
    if rank == 0:
        return None  # General manager process does not have a stage
    return (rank - 1) // pipeline_size

def get_stage_in_pipeline(rank, pipeline_size=4):
    """
    Given a worker rank, returns the stage number within its pipeline.

    :param rank: process rank (manager is rank 0)
    :param pipeline_size: number of workers per pipeline
    :return: stage number (with possible values 1 to pipeline_size)
    """
    if rank == 0:
        return None  # General manager process does not have a stage
    return ((rank - 1) % pipeline_size) + 1

def get_pipeline_prev_next(rank, pipeline_size=4):
    """
    Returns the previous and next rank in the same pipeline for a given worker.

    :param rank: process rank (manager is rank 0)
    :param pipeline_size: number of workers per pipeline
    :return: tuple (prev_rank, next_rank), None if no prev or next (each tuple element is from 1 to pipeline_size)
    """
    stage = get_stage_in_pipeline(rank, pipeline_size)
    prev_rank = rank - 1 if stage > 1 else None
    next_rank = rank + 1 if stage < pipeline_size else None
    return prev_rank, next_rank

def get_pipeline_last_worker_rank(pipeline_id, pipeline_size=4):
    """
    Given a pipeline ID, returns the rank of the last worker in that pipeline.

    :param pipeline_id: 0-based pipeline ID
    :param pipeline_size: number of workers per pipeline
    :return: rank of the last worker in the pipeline
    """
    return 1 + (pipeline_id * pipeline_size) + (pipeline_size-1)

def get_first_worker_rank(pipeline_id, pipeline_size=4):
    """
    Given a pipeline ID, returns the rank of the first worker in that pipeline.

    :param pipeline_id: 0-based pipeline ID
    :param pipeline_size: number of workers per pipeline
    :return: rank of the first worker in the pipeline
    """
    return 1 + (pipeline_id * pipeline_size)

def get_worker_pair(rank):
    """
    given a worker rank (starting from 1), returns its paired worker
    pairing is 1<->2, 3<->4, 5<->6, ...
    :param rank: worker rank
    :return: rank of partner worker
    """
    if rank % 2 == 1:   # odd rank
        return rank + 1
    else:                # even rank
        return rank - 1

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

def pattern3(comm, rank, size, text_lines, vocab_set, stopwords_set):
    """
    Pattern #3: Parallel Pipelines (Multiple Independent Linear Pipelines)
    Each pipeline has 4 workers performing stages as in Pattern #2:
    W1(Lower) -> W2(Punct) -> W3(Stop) -> W4(TF)
    Only term-frequency is computed.
    Number of pipelines is deduced from total processes: num_pipelines = (size - 1) // 4
    :param comm: MPI communicator
    :param rank: rank of the processor
    :param size: size of the processor
    :param text_lines: input text to be processed
    :param vocab_set: vocabulary set
    :param stopwords_set: commonly used words such as “the”, “is”, “to”, and “in”, which are called stopwords
    :return: tf_count (if manager aka rank=0)
    """
    pipeline_size = 4
    num_pipelines = (size - 1) // pipeline_size
    manager_rank = 0

    if rank == manager_rank:   # manager process
        # send config info to all processes except the manager process
        for r in range(1, size):
            comm.send((vocab_set, stopwords_set), dest=r)

        # divide data into some large chunks to send to each pipeline
        large_chunks = split_into_chunks(text_lines, num_pipelines)
        final_tf = Counter()

        # for each pipeline, create smaller chunk and send to first worker
        for pipeline_id, large_chunk in enumerate(large_chunks):
            divisor = 10
            num_small_chunks = len(large_chunk)//divisor
            small_chunks = split_into_chunks(large_chunk, num_small_chunks)
            first_worker_rank = get_first_worker_rank(pipeline_id, pipeline_size)
            for chunk in small_chunks:
                comm.send(chunk, dest=first_worker_rank)
            
            # send none data to give the info whole data is sent
            comm.send(None, dest=first_worker_rank)

        # increase aggregated final tf value from the values from each pipeline
        for pipeline_id in range(num_pipelines):
            last_worker_rank = get_pipeline_last_worker_rank(pipeline_id, pipeline_size)
            partial_tf = comm.recv(source=last_worker_rank)
            final_tf.update(partial_tf)

        return final_tf, None

    else:   # worker process
        # Receive pipeline and stage info
        pipeline_id = get_pipeline_id(rank, pipeline_size)
        stage = get_stage_in_pipeline(rank, pipeline_size)
        prev_rank, next_rank = get_pipeline_prev_next(rank, pipeline_size)

        # get config from manager process
        vocab, stopwords = comm.recv(source=manager_rank)

        # if the worker is the last stage init a final counter for this pipeline
        if stage == pipeline_size:
            acc_tf = Counter()
        else:
            acc_tf = None

        while True:
            # get data from prev stage in the same pipeline
            source = prev_rank if prev_rank is not None else 0
            data = comm.recv(source=source)


            # if no data send, then we are done
            if data is None:
                if stage < pipeline_size:
                    # send we're done to next worker
                    comm.send(None, dest=next_rank)
                else:
                    # if the last worker send the counter to manager
                    comm.send(acc_tf, dest=manager_rank)
                break

            # pipelining logic
            processed_chunk = []
            
            if stage == 1:  # W1 lower case operation
                for line in data:
                    processed_chunk.append(line.lower())
                if stage < pipeline_size:
                    comm.send(processed_chunk, dest=next_rank)
            
            elif stage == 2:  # W2 punctuation removal
                for line in data:
                    clean_line = ""
                    for char in line:
                        if char not in string.punctuation:
                            clean_line += char
                    processed_chunk.append(clean_line)
                if stage < pipeline_size:
                    comm.send(processed_chunk, dest=next_rank)
            
            elif stage == 3:  # W3 stopword removal
                for line in data:
                    words = line.split()
                    clean = [w for w in words if w not in stopwords]
                    processed_chunk.append(clean)
                if stage < pipeline_size:
                    comm.send(processed_chunk, dest=next_rank)
            
            elif stage == 4:  # W4 Term Frequency (TF)
                partial_tf = compute_tf(data, vocab)
                acc_tf.update(partial_tf)

        return None, None

def pattern4(comm, rank, size, text_lines, vocab_set, stopwords_set):
    """
    pattern #4: end-to-end processing in worker processes with task paralellism
    each worker does lowercase, punctuation removal, stopword removal
    then we do tf/df splitting: odd rank -> tf, even rank -> df
    manager collects and aggregates
    :param comm: mpi communicator
    :param rank: rank of processor
    :param size: number of processors
    :param text_lines: input text
    :param vocab_set: vocabulary set
    :param stopwords_set: stopwords set
    :return: final_tf, final_df (only for manager)
    """

    manager_rank = 0
    num_workers = size - 1  # exclude manager

    if rank == manager_rank:   # manager process
        # send config info to all processes except the manager process
        for r in range(1, size):
            comm.send((vocab_set, stopwords_set), dest=r)

        # split data into balanced chunks for workers
        chunks = split_into_chunks(text_lines, num_workers)
        
        # send each chunk to each worker
        for i, chunk in enumerate(chunks):
            comm.send(chunk, dest=i+1)
        
        # collect results from all workers
        final_tf = Counter()
        final_df = Counter()
        for worker_rank in range(1, size):
            tf_part, df_part = comm.recv(source=worker_rank)
            if tf_part:
                final_tf.update(tf_part)
            if df_part:
                final_df.update(df_part)
        return final_tf, final_df

    else:   # worker process
        # get config from manager process
        vocab, stopwords = comm.recv(source=manager_rank)

        # receive data from manager
        data_chunk = comm.recv(source=manager_rank)
        
        # preprocess in each worker
        preprocessed_data_chunk = []
        for line in data_chunk:
            line = line.lower() # make lowercase
            line = "".join(c for c in line if c not in string.punctuation)  # remove punctuation
            words = line.split() 
            clean_words = [w for w in words if w not in stopwords] # remove stopwords
            preprocessed_data_chunk.append(clean_words)

        pair_rank = get_worker_pair(rank)

        # asymmetric send/recv to avoid deadlocks
        if rank % 2 == 1:  # odd rank sends first
            comm.send(preprocessed_data_chunk, dest=pair_rank)
            received_chunk = comm.recv(source=pair_rank)
        else:  # even rank receives first
            received_chunk = comm.recv(source=pair_rank)
            comm.send(preprocessed_data_chunk, dest=pair_rank)

        # combine own and partner's data
        combined_chunk = preprocessed_data_chunk + received_chunk

        # calculate tf or df depending on the modulo of rank, if odd: tf, if even: df
        tf_count = compute_tf(combined_chunk, vocab_set) if rank % 2 == 1 else None
        df_count = compute_df(combined_chunk, vocab_set) if rank % 2 == 0 else None

        # send results back to manager process
        comm.send((tf_count, df_count), dest=manager_rank)
        return None, None

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
        final_tf, final_df = pattern3(comm, rank, size, text_lines, vocab_set, stopwords_set)
    elif pattern == 4:
        final_tf, final_df = pattern4(comm, rank, size, text_lines, vocab_set, stopwords_set)
    
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
