from collections import defaultdict
import random
import json
from retvec import RecVec
from retvec.utils import tf_cap_memory, find_primes
from tqdm.auto import tqdm
from time import time
import tensorflow as tf
from copy import copy
from termcolor import cprint

import logging

tf.get_logger().setLevel(logging.ERROR)
tf_cap_memory()

REVIVE = 20  # revive the best list every N population
NUM_TRIALS = 5000
MAX_PATIENCE = 30  # How many time before reset after improvement
NUM_SMALL_PRIMES_GEN = 10  # how many initial generation using only "small primes"

MAX_PRIME = 500000  # wider
MAX_SMALL_PRIMES = 3000  # initial step
MAX_START_PRIMES = 150  # used to see the new list

TARGET_PRIMES = reversed(list(range(4, 17)))  # how many primes to use

EMBEDDING_SIZES = [24, 32, 40, 48]  # 16

ALPHABET_SIZE = 55000
START_LISTS = {
    "16": [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 71, 73, 79],
    "24": [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
    "32": [37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101],
    "40": [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103],
    "48": [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]
}

# load primes

PRIMES = [int(i) for i in find_primes(MAX_PRIME)]
SMALL_PRIMES = [int(i) for i in find_primes(MAX_SMALL_PRIMES)]
SMALL_PRIMES = SMALL_PRIMES[6:]

START_PRIMES = [int(i) for i in find_primes(MAX_START_PRIMES)]
START_PRIMES = START_PRIMES[6:]

print(len(PRIMES), PRIMES[:100])

chars = [chr(i) for i in range(ALPHABET_SIZE)]

# do a few permutations even though the positional embedding should not be an issue
batch = []
for i in range(32):
    random.shuffle(chars)
    s = "".join(chars)
batch.append([s])

ts = int(time())
log = open('data/prime_search_log.%d.json' % ts, 'a+')
best = open('data/prime_search_best.%d.json' % ts, 'a+')
for NUM_PRIMES in TARGET_PRIMES:
    for EMBEDDING_SIZE in EMBEDDING_SIZES:
        min_collision = ALPHABET_SIZE
        populations = 0
        patience = 0
        best_list = []
        cur_lowest_collisions = ALPHABET_SIZE
        prime_list = START_LISTS["%s" % EMBEDDING_SIZE][:NUM_PRIMES]
        curr_best_list = prime_list
        generations = 0
        best_generation = 0
        improvement = 0
        pb = tqdm(total=NUM_TRIALS)
        for tries in range(NUM_TRIALS):

            # kill population that don't improve.
            if patience == MAX_PATIENCE:
                info = {
                    "best_generation": best_generation,
                    "collisions": cur_lowest_collisions,
                    "improvement_steps": improvement,
                    "list": curr_best_list,
                    "embedding": EMBEDDING_SIZE,
                    "alphabet": ALPHABET_SIZE,
                    "num_primes": NUM_PRIMES
                }
                log.write("%s\n" % json.dumps(info))
                log.flush()
                improvement = 0
                patience = 0
                generations = 0
                best_generation = 0
                populations += 1
                cur_lowest_collisions = ALPHABET_SIZE

                # revive the best list periodically
                if populations % REVIVE == 0:
                    prime_list = copy(best_list)
                    cprint("\n Reviving best list: %s" % best_list, 'yellow')
                else:
                    old_list = copy(prime_list)
                    prime_list = random.sample(START_PRIMES, NUM_PRIMES)
                    cprint('\nChange %s -> %s' % (old_list, prime_list),
                           'cyan')
                curr_best_list = copy(prime_list)
                # clear stuff
                tf.keras.backend.clear_session()

            curr_collisions = 0
            for s in batch:
                ce = RecVec(max_len=ALPHABET_SIZE,
                            embedding_size=EMBEDDING_SIZE,
                            positional_encoding=True,
                            primes=prime_list,
                            is_eager=True)
                embeddings = ce(s)
                collisions = defaultdict(int)
                npe = embeddings[0].numpy()
                for idx in range(ALPHABET_SIZE):
                    v = "%s" % list(npe[idx])
                    collisions[v] += 1
                for v in collisions.values():
                    curr_collisions += v - 1

                ce = RecVec(max_len=ALPHABET_SIZE,
                            embedding_size=EMBEDDING_SIZE,
                            primes=prime_list,
                            positional_encoding=False,
                            is_eager=True)
                embeddings = ce(s)
                collisions = defaultdict(int)
                npe = embeddings[0].numpy()
                for idx in range(ALPHABET_SIZE):
                    v = "%s" % list(npe[idx])
                    collisions[v] += 1
                for v in collisions.values():
                    curr_collisions += v - 1

            if curr_collisions < min_collision:
                min_collision = curr_collisions
                best_list = prime_list
                patience = 0
                cprint(
                    "\nNew Best: %d collisions: %s" %
                    (min_collision, best_list), 'magenta')
                if not min_collision:

                    info = {
                        "best_generation": best_generation,
                        "collisions": min_collision,
                        "improvement_steps": improvement,
                        "list": best_list,
                        "embedding": EMBEDDING_SIZE,
                        "alphabet": ALPHABET_SIZE,
                        "num_primes": NUM_PRIMES
                    }
                    log.write("%s\n" % json.dumps(info))
                    log.flush()

                    print("perfect embedding achieved! breaking", 'magenta')
                    break

            if curr_collisions < cur_lowest_collisions:
                curr_best_list = copy(prime_list)
                cur_lowest_collisions = curr_collisions
                best_generation = generations
                improvement += 1
            else:
                # reset if no improvement
                prime_list = copy(curr_best_list)

            # select new prime
            if generations < NUM_SMALL_PRIMES_GEN:
                new_prime = random.sample(SMALL_PRIMES, 1)
            else:
                new_prime = random.sample(PRIMES, 1)

            pos_prim = random.randint(0, len(prime_list) - 1)
            prime_list[pos_prim] = new_prime[0]

            patience += 1
            generations += 1
            pb.update()
            pb.set_postfix({
                "lowest": min_collision,
                "last": curr_collisions,
                "lowest_curr": cur_lowest_collisions,
                "improved_steps": improvement,
                "populations": populations,
                'patience': patience,
                'generations': generations,
                'primes': NUM_PRIMES,
                'emb_size': EMBEDDING_SIZE
            })

        pb.close()

        # write the best
        info = {
            "collisions": min_collision,
            "list": best_list,
            "embedding": EMBEDDING_SIZE,
            "alphabet": ALPHABET_SIZE,
            "num_primes": NUM_PRIMES
        }
        best.write("%s\n" % json.dumps(info))
        best.flush()
