###
# This files just contains the sequence of envs to be used
#
#
###

GOOD_RPO_ENVS = [
    "door-close-v3",
    "sweep-into-v3",
    "coffee-button-v3",
    "window-open-v3",
    "reach-wall-v3",
    "drawer-close-v3",
    "button-press-v3",
    "plate-slide-back-side-v3",
    "reach-v3",
    "plate-slide-side-v3",
    "coffee-push-v3",
    "plate-slide-back-v3",
    "soccer-v3",
    "window-close-v3",
    "handle-pull-side-v3",
    "hand-insert-v3",
    "door-lock-v3",
    "push-v3",
    "peg-unplug-side-v3"
]

## Removing the following tasks since they seem too difficult
# 'lever_pull', 'peg-insert-side', 'button-press-topdown', 'button-press-wall',
# 'stick-push', 'coffee-pull', 'push-wall'
## Removing more
# "button-press-topdown-wall", "plate-slide-v3", "handle-press-side", "handle-press"
# currently 20 left


# randomly generated below
# we keep these fixed so the agents have a fair comparison

RPO10_SEQ = [['handle-pull-side-v3', 'peg-unplug-side-v3', 'coffee-push-v3', 'soccer-v3', 'drawer-close-v3', 'reach-wall-v3', 'plate-slide-back-v3', 'window-open-v3', 'plate-slide-side-v3', 'plate-slide-back-side-v3'],
             ['window-close-v3', 'window-open-v3', 'hand-insert-v3', 'door-lock-v3', 'reach-v3', 'button-press-v3', 'sweep-into-v3', 'coffee-button-v3', 'door-close-v3', 'push-v3'],
             ['window-close-v3', 'reach-wall-v3', 'sweep-into-v3', 'reach-v3', 'soccer-v3', 'coffee-push-v3', 'plate-slide-side-v3', 'drawer-close-v3', 'hand-insert-v3', 'door-close-v3'],
             ['plate-slide-back-v3', 'reach-wall-v3', 'door-lock-v3', 'peg-unplug-side-v3', 'push-v3', 'button-press-v3', 'plate-slide-back-side-v3', 'coffee-push-v3', 'coffee-button-v3', 'handle-pull-side-v3'],
             ['push-v3', 'coffee-button-v3', 'sweep-into-v3', 'door-close-v3', 'drawer-close-v3', 'soccer-v3', 'peg-unplug-side-v3', 'hand-insert-v3', 'door-lock-v3', 'reach-v3'],
             ['button-press-v3', 'plate-slide-back-side-v3', 'window-close-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3', 'plate-slide-back-v3', 'coffee-button-v3', 'window-open-v3', 'handle-pull-side-v3', 'door-close-v3'],
             ['push-v3', 'button-press-v3', 'plate-slide-back-v3', 'drawer-close-v3', 'soccer-v3', 'plate-slide-side-v3', 'reach-wall-v3', 'coffee-push-v3', 'window-close-v3', 'door-lock-v3'],
             ['plate-slide-side-v3', 'hand-insert-v3', 'handle-pull-side-v3', 'plate-slide-back-side-v3', 'window-open-v3', 'sweep-into-v3', 'reach-wall-v3', 'reach-v3', 'soccer-v3', 'peg-unplug-side-v3'],
             ['hand-insert-v3', 'reach-v3', 'window-close-v3', 'drawer-close-v3', 'window-open-v3', 'coffee-button-v3', 'plate-slide-back-v3', 'coffee-push-v3', 'push-v3', 'plate-slide-back-side-v3'],
             ['sweep-into-v3', 'peg-unplug-side-v3', 'window-close-v3', 'door-lock-v3', 'hand-insert-v3', 'handle-pull-side-v3', 'window-open-v3', 'door-close-v3', 'button-press-v3', 'reach-wall-v3'],
             ['reach-v3', 'door-lock-v3', 'sweep-into-v3', 'push-v3', 'button-press-v3', 'coffee-push-v3', 'handle-pull-side-v3', 'plate-slide-side-v3', 'door-close-v3', 'drawer-close-v3'],
             ['plate-slide-back-side-v3', 'soccer-v3', 'sweep-into-v3', 'handle-pull-side-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3', 'door-lock-v3', 'reach-v3', 'plate-slide-back-v3', 'coffee-button-v3'],
             ['reach-wall-v3', 'plate-slide-back-v3', 'drawer-close-v3', 'hand-insert-v3', 'coffee-push-v3', 'coffee-button-v3', 'window-close-v3', 'plate-slide-back-side-v3', 'door-close-v3', 'button-press-v3'],
             ['soccer-v3', 'drawer-close-v3', 'push-v3', 'sweep-into-v3', 'window-open-v3', 'reach-wall-v3', 'door-lock-v3', 'window-close-v3', 'reach-v3', 'hand-insert-v3'],
             ['plate-slide-back-v3', 'plate-slide-side-v3', 'door-close-v3', 'push-v3', 'peg-unplug-side-v3', 'plate-slide-back-side-v3', 'coffee-push-v3', 'coffee-button-v3', 'button-press-v3', 'soccer-v3'],
             ['hand-insert-v3', 'coffee-button-v3', 'soccer-v3', 'window-open-v3', 'push-v3', 'reach-v3', 'drawer-close-v3', 'handle-pull-side-v3', 'door-lock-v3', 'plate-slide-back-side-v3'],
             ['coffee-push-v3', 'door-close-v3', 'handle-pull-side-v3', 'window-close-v3', 'plate-slide-back-v3', 'reach-wall-v3', 'sweep-into-v3', 'window-open-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3'],
             ['coffee-push-v3', 'button-press-v3', 'reach-v3', 'peg-unplug-side-v3', 'reach-wall-v3', 'door-close-v3', 'window-open-v3', 'handle-pull-side-v3', 'plate-slide-back-side-v3', 'soccer-v3'],
             ['sweep-into-v3', 'plate-slide-side-v3', 'button-press-v3', 'drawer-close-v3', 'push-v3', 'coffee-button-v3', 'door-lock-v3', 'hand-insert-v3', 'plate-slide-back-v3', 'window-close-v3'],
             ['reach-v3', 'button-press-v3', 'plate-slide-side-v3', 'door-close-v3', 'plate-slide-back-side-v3', 'plate-slide-back-v3', 'coffee-button-v3', 'sweep-into-v3', 'reach-wall-v3', 'drawer-close-v3']]


# Shuffled order of RPO10_SEQ[19] (the "set0" sequence used in the main run),
# for the task-order robustness check. Same 10 tasks as set0, only the order
# differs. Fixed permutation numpy.default_rng(42).permutation(10) so every
# method and seed sees the same order.
SHUFFLED_SET0_SEQ = ['plate-slide-back-v3', 'coffee-button-v3', 'reach-v3', 'sweep-into-v3', 'door-close-v3', 'plate-slide-side-v3', 'plate-slide-back-side-v3', 'drawer-close-v3', 'button-press-v3', 'reach-wall-v3']


RPO10_SEQ_OLD = [['plate-slide-back-v3', 'button-press-v3', 'handle-pull-side-v3', 'peg-unplug-side-v3', 'window-open-v3', 'handle-pull-v3', 'plate-slide-back-side-v3', 'coffee-push-v3', 'push-v3', 'reach-v3'],
             ['door-close-v3', 'hand-insert-v3', 'window-close-v3', 'sweep-into-v3', 'drawer-close-v3', 'plate-slide-side-v3', 'soccer-v3', 'reach-wall-v3', 'door-lock-v3', 'coffee-button-v3'],
             ['sweep-into-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3', 'window-open-v3', 'door-lock-v3', 'reach-v3', 'reach-wall-v3', 'plate-slide-back-v3', 'door-close-v3', 'drawer-close-v3'],
             ['handle-pull-v3', 'soccer-v3', 'handle-pull-side-v3', 'button-press-v3', 'push-v3', 'window-close-v3', 'coffee-button-v3', 'hand-insert-v3', 'plate-slide-back-side-v3', 'coffee-push-v3'],
             ['window-open-v3', 'push-v3', 'reach-v3', 'plate-slide-back-v3', 'handle-pull-side-v3', 'soccer-v3', 'sweep-into-v3', 'coffee-button-v3', 'window-close-v3', 'coffee-push-v3'],
             ['drawer-close-v3', 'peg-unplug-side-v3', 'plate-slide-side-v3', 'door-lock-v3', 'plate-slide-back-side-v3', 'reach-wall-v3', 'handle-pull-v3', 'button-press-v3', 'hand-insert-v3', 'door-close-v3'],
             ['door-lock-v3', 'coffee-button-v3', 'hand-insert-v3', 'window-close-v3', 'button-press-v3', 'plate-slide-back-side-v3', 'plate-slide-back-v3', 'window-open-v3', 'reach-wall-v3', 'soccer-v3'],
             ['drawer-close-v3', 'handle-pull-v3', 'push-v3', 'plate-slide-side-v3', 'coffee-push-v3', 'handle-pull-side-v3', 'sweep-into-v3', 'reach-v3', 'peg-unplug-side-v3', 'door-close-v3'],
             ['hand-insert-v3', 'door-lock-v3', 'window-close-v3', 'reach-wall-v3', 'coffee-push-v3', 'drawer-close-v3', 'plate-slide-back-v3', 'handle-pull-v3', 'button-press-v3', 'sweep-into-v3'],
             ['reach-v3', 'push-v3', 'plate-slide-back-side-v3', 'plate-slide-side-v3', 'door-close-v3', 'soccer-v3', 'peg-unplug-side-v3', 'window-open-v3', 'handle-pull-side-v3', 'coffee-button-v3']]


RPO20_SEQ = [
['door-lock-v3', 'handle-press-v3', 'handle-press-side-v3', 'button-press-v3', 'door-close-v3', 'hand-insert-v3', 'reach-v3', 'plate-slide-v3', 'handle-pull-side-v3', 'plate-slide-back-side-v3', 'plate-slide-back-v3', 'soccer-v3', 'sweep-into-v3', 'reach-wall-v3', 'window-open-v3', 'coffee-button-v3', 'coffee-push-v3', 'peg-unplug-side-v3', 'window-close-v3', 'plate-slide-side-v3'],
['plate-slide-side-v3', 'plate-slide-back-v3', 'handle-press-side-v3', 'peg-unplug-side-v3', 'handle-pull-v3', 'reach-wall-v3', 'plate-slide-back-side-v3', 'button-press-v3', 'soccer-v3', 'hand-insert-v3', 'door-lock-v3', 'push-v3', 'window-close-v3', 'button-press-topdown-wall-v3', 'drawer-close-v3', 'sweep-into-v3', 'reach-v3', 'coffee-button-v3', 'coffee-push-v3', 'door-close-v3'],
['hand-insert-v3', 'peg-unplug-side-v3', 'handle-pull-side-v3', 'handle-press-v3', 'button-press-v3', 'coffee-push-v3', 'plate-slide-back-v3', 'handle-pull-v3', 'button-press-topdown-wall-v3', 'push-v3', 'plate-slide-v3', 'door-close-v3', 'reach-v3', 'window-open-v3', 'coffee-button-v3', 'window-close-v3', 'drawer-close-v3', 'soccer-v3', 'plate-slide-side-v3', 'plate-slide-back-side-v3'],
['handle-pull-side-v3', 'button-press-v3', 'window-open-v3', 'door-close-v3', 'reach-wall-v3', 'push-v3', 'hand-insert-v3', 'drawer-close-v3', 'handle-press-side-v3', 'handle-press-v3', 'door-lock-v3', 'plate-slide-back-side-v3', 'window-close-v3', 'sweep-into-v3', 'button-press-topdown-wall-v3', 'coffee-button-v3', 'soccer-v3', 'handle-pull-v3', 'plate-slide-back-v3', 'plate-slide-v3'],
['soccer-v3', 'coffee-button-v3', 'handle-press-v3', 'handle-pull-side-v3', 'plate-slide-back-v3', 'door-lock-v3', 'drawer-close-v3', 'reach-v3', 'peg-unplug-side-v3', 'plate-slide-v3', 'reach-wall-v3', 'handle-pull-v3', 'push-v3', 'plate-slide-side-v3', 'coffee-push-v3', 'button-press-topdown-wall-v3', 'hand-insert-v3', 'sweep-into-v3', 'window-open-v3', 'handle-press-side-v3'],
['door-close-v3', 'reach-wall-v3', 'coffee-push-v3', 'sweep-into-v3', 'door-lock-v3', 'plate-slide-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3', 'handle-press-side-v3', 'handle-pull-v3', 'window-close-v3', 'push-v3', 'plate-slide-back-side-v3', 'reach-v3', 'handle-press-v3', 'window-open-v3', 'button-press-topdown-wall-v3', 'handle-pull-side-v3', 'drawer-close-v3', 'button-press-v3'],
['window-open-v3', 'window-close-v3', 'handle-pull-v3', 'push-v3', 'plate-slide-back-v3', 'button-press-v3', 'reach-v3', 'plate-slide-v3', 'coffee-button-v3', 'handle-pull-side-v3', 'hand-insert-v3', 'reach-wall-v3', 'drawer-close-v3', 'plate-slide-back-side-v3', 'sweep-into-v3', 'button-press-topdown-wall-v3', 'door-close-v3', 'plate-slide-side-v3', 'door-lock-v3', 'coffee-push-v3'],
['door-close-v3', 'soccer-v3', 'drawer-close-v3', 'handle-pull-side-v3', 'plate-slide-back-v3', 'hand-insert-v3', 'coffee-push-v3', 'reach-wall-v3', 'peg-unplug-side-v3', 'button-press-topdown-wall-v3', 'plate-slide-side-v3', 'reach-v3', 'window-close-v3', 'sweep-into-v3', 'button-press-v3', 'coffee-button-v3', 'handle-press-side-v3', 'handle-pull-v3', 'window-open-v3', 'handle-press-v3'],
['handle-pull-side-v3', 'push-v3', 'plate-slide-back-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3', 'reach-wall-v3', 'sweep-into-v3', 'door-lock-v3', 'plate-slide-v3', 'window-open-v3', 'handle-press-v3', 'hand-insert-v3', 'handle-press-side-v3', 'handle-pull-v3', 'soccer-v3', 'drawer-close-v3', 'reach-v3', 'button-press-v3', 'window-close-v3', 'plate-slide-back-side-v3'],
['peg-unplug-side-v3', 'handle-press-side-v3', 'reach-wall-v3', 'door-close-v3', 'button-press-topdown-wall-v3', 'reach-v3', 'handle-pull-v3', 'drawer-close-v3', 'plate-slide-side-v3', 'coffee-button-v3', 'window-close-v3', 'handle-press-v3', 'door-lock-v3', 'coffee-push-v3', 'window-open-v3', 'plate-slide-back-side-v3', 'button-press-v3', 'push-v3', 'plate-slide-v3', 'soccer-v3']
]





GOOD_RPO_SEQS = [
    ['sweep-into-v3', 'door-close-v3', 'drawer-close-v3', 'button-press-v3', 'window-close-v3', 'hand-insert-v3', 'soccer-v3', 'handle-pull-side-v3'],
    ['peg-unplug-side-v3', 'soccer-v3', 'plate-slide-side-v3', 'plate-slide-back-side-v3', 'reach-wall-v3', 'reach-v3', 'drawer-close-v3', 'coffee-button-v3'],
    ['window-close-v3', 'window-open-v3', 'button-press-topdown-wall-v3', 'handle-press-v3', 'coffee-button-v3', 'handle-press-side-v3', 'push-v3', 'plate-slide-back-v3'],
    ['plate-slide-v3', 'door-close-v3', 'sweep-into-v3', 'door-lock-v3', 'coffee-push-v3', 'handle-press-v3', 'hand-insert-v3', 'reach-wall-v3'],
    ['coffee-button-v3', 'soccer-v3', 'plate-slide-v3', 'handle-pull-v3', 'window-open-v3', 'handle-press-v3', 'handle-pull-side-v3', 'drawer-close-v3'],
    ['handle-pull-side-v3', 'handle-press-v3', 'window-open-v3', 'handle-pull-v3', 'window-close-v3', 'coffee-button-v3', 'plate-slide-v3', 'coffee-push-v3'],
    ['button-press-v3', 'button-press-topdown-wall-v3', 'handle-press-side-v3', 'reach-wall-v3', 'hand-insert-v3', 'plate-slide-side-v3', 'peg-unplug-side-v3', 'push-v3'],
    ['peg-unplug-side-v3', 'reach-wall-v3', 'plate-slide-side-v3', 'door-close-v3', 'door-lock-v3', 'handle-press-side-v3', 'reach-v3', 'plate-slide-back-side-v3'],
    ['coffee-push-v3', 'handle-pull-v3', 'sweep-into-v3', 'plate-slide-back-v3', 'plate-slide-back-side-v3', 'reach-v3', 'plate-slide-v3', 'window-open-v3'],
    ['button-press-topdown-wall-v3', 'push-v3', 'drawer-close-v3', 'door-lock-v3', 'plate-slide-back-v3', 'door-close-v3', 'sweep-into-v3', 'button-press-v3']
    ]

## OLD LIST 1st attempt
# GOOD_RPO_SEQS = [['plate-slide-back-v3', 'button-press-v3', 'door-close-v3', 'handle-press-v3', 'button-press-topdown-v3', 'plate-slide-v3', 'lever-pull-v3', 'peg-insert-side-v3'],
#                  ['plate-slide-side-v3', 'lever-pull-v3', 'window-close-v3', 'drawer-close-v3', 'soccer-v3', 'sweep-into-v3', 'button-press-wall-v3', 'handle-press-v3'],
#                  ['button-press-wall-v3', 'handle-press-side-v3', 'coffee-push-v3', 'peg-unplug-side-v3', 'soccer-v3', 'button-press-v3', 'button-press-topdown-wall-v3', 'platex`-slide-v3'],
#                  ['reach-v3', 'push-wall-v3', 'coffee-button-v3', 'plate-slide-back-side-v3', 'stick-push-v3', 'plate-slide-v3', 'coffee-pull-v3', 'handle-press-v3'],
#                  ['plate-slide-back-side-v3', 'push-v3', 'peg-insert-side-v3', 'reach-wall-v3', 'sweep-into-v3', 'button-press-wall-v3', 'handle-pull-side-v3', 'coffee-button-v3'],
#                  ['stick-push-v3', 'reach-wall-v3', 'peg-insert-side-v3', 'coffee-pull-v3', 'handle-pull-v3', 'coffee-push-v3', 'door-lock-v3', 'plate-slide-back-side-v3'],
#                  ['window-open-v3', 'peg-unplug-side-v3', 'hand-insert-v3', 'door-lock-v3', 'button-press-v3', 'handle-pull-v3', 'reach-v3', 'handle-press-side-v3'],
#                  ['coffee-button-v3', 'plate-slide-side-v3', 'door-close-v3', 'drawer-close-v3', 'window-close-v3', 'plate-slide-back-v3', 'button-press-topdown-v3', 'button-press-topdown-wall-v3'],
#                  ['reach-v3', 'push-wall-v3', 'drawer-close-v3', 'hand-insert-v3', 'window-open-v3', 'handle-press-side-v3', 'button-press-topdown-v3', 'door-close-v3'],
#                  ['coffee-push-v3', 'window-open-v3', 'sweep-into-v3', 'handle-pull-side-v3', 'reach-wall-v3', 'plate-slide-side-v3', 'push-v3', 'plate-slide-back-v3']]


if __name__ == "__main__":
    # for i, x in enumerate(RPO20_SEQ[9]):
    #     print(i+1, x)
    # quit()

    import random
    import numpy as np
    NUM_TASKS_PER_SEQ = 10
    NUM_SEQ = 20
    total_tasks = NUM_TASKS_PER_SEQ * NUM_SEQ

    #### Let's use stratified sampling, we make sure all tasks are represented as evenly as possible

    ### Method based on choosing tasks first, then adding the unused ones to the next set
    all_seq_tasks = []


    candidate_set = set(GOOD_RPO_ENVS)

    # we assume that the initial candidate set is larger than then number of tasks to sample
    for i_seq in range(NUM_SEQ):
        one_set_tasks = set([])
        num_tasks_to_sample = NUM_TASKS_PER_SEQ

        if len(candidate_set) < NUM_TASKS_PER_SEQ:
            one_set_tasks = one_set_tasks.union(candidate_set)
            num_tasks_to_sample -= len(candidate_set)
            candidate_set = set(GOOD_RPO_ENVS)

        # sample from the candidates ensuring there's no repeats
        sampled_set_tasks = random.sample(list(candidate_set.difference(one_set_tasks)), num_tasks_to_sample)
        one_set_tasks = one_set_tasks.union(sampled_set_tasks)

        # remove the sampled tasks from candidates
        candidate_set = candidate_set.difference(sampled_set_tasks)

        # add to master list of sequences
        all_seq_tasks.append(one_set_tasks)


    # make tasks into lists and shuffle them
    all_seq_tasks = [list(task_set) for task_set in all_seq_tasks]
    for task_lst in all_seq_tasks:
        random.shuffle(task_lst)

    # print them
    print(all_seq_tasks)






    ### Method based on rejection-sampling all the tasks and then regrouping them
    # seq_list = GOOD_RPO_ENVS * (int(total_tasks / len(GOOD_RPO_ENVS)) + 1)
    # seq_list = seq_list[:total_tasks]
    #
    #
    #
    # # printl(seq_list)z
    # import collections
    #
    # def check_duplicate(lst):
    #     counter = collections.Counter(lst)
    #     if counter.most_common(1)[0][1] > 1:  # check count of most common item
    #         return True
    #     return False
    #
    # same_task_twice = True
    #
    # # do rejection sampling until you find an ordering where tasks aren't duplicated in the same sequence
    # i_try = 0
    # while same_task_twice:
    #     same_task_twice = False
    #     random.shuffle(seq_list)
    #     if np.any([check_duplicate(seq_list[i*NUM_TASKS_PER_SEQ:(i+1)*NUM_TASKS_PER_SEQ]) for i in range(NUM_SEQ)]):
    #         same_task_twice = True
    #     i_try += 1
    #     if i_try % 100 == 0:
    #         print(i_try)
    #
    # # print(seq_list)
    #
    # seq_list = [ seq_list[i*NUM_TASKS_PER_SEQ:(i+1)*NUM_TASKS_PER_SEQ] for i in range(NUM_SEQ)]
    # for lst in seq_list:
    #     print(lst)
    #
    # ######
    # # for i in range(10):
    # #     rand_list = random.sample(GOOD_RPO_ENVS, 8)
    # #     seq_list.append(rand_list)
    # # print(seq_list)
    # # len(GOOD_RPO_ENVS)
    #
    # # printl(seq_list)z
    # import collections
    #
    # def check_duplicate(lst):
    #     counter = collections.Counter(lst)
    #     if counter.most_common(1)[0][1] > 1:  # check count of most common item
    #         return True
    #     return False
    #
    # same_task_twice = True
    #
    # # do rejection sampling until you find an ordering where tasks aren't duplicated in the same sequence
    # i_try = 0
    # while same_task_twice:
    #     same_task_twice = False
    #     random.shuffle(seq_list)
    #     if np.any([check_duplicate(seq_list[i*8:(i+1)*8]) for i in range(10)]):
    #         same_task_twice = True
    #     i_try += 1
    #     if i_try % 10000 == 0:
    #         print(i_try)
    #
    # # print(seq_list)
    #
    # seq_list = [ seq_list[i*8:(i+1)*8] for i in range(10)]
    # for lst in seq_list:
    #     print(lst)

    ######
    # for i in range(10):
    #     rand_list = random.sample(GOOD_RPO_ENVS, 8)
    #     seq_list.append(rand_list)
    # print(seq_list)
    #


    ## Checking distribution of tasks
    import collections
    seq_list = RPO10_SEQ

    for lst in seq_list:
        print(len(lst))

    flat_list = [
        x
        for xs in seq_list
        for x in xs
    ]
    # for env in GOOD_RPO_ENVS:
    #     print(env in flat_list, env)

    counter = collections.Counter(flat_list)
    print(counter)