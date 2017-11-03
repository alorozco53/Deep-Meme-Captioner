#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import configuration as config
import argparse
import time
import os
import math

from model import MemeModel

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Training script\'s argument parser')
parser.add_argument('checkpoint_dir', help='directory containing the model checkpoints')
parser.add_argument('eval_dir', help='directory to write event logs')
parser.add_argument('--eval_interval_secs', default=600, help='interval between evaluation runs', type=int)
parser.add_argument('--num_eval_examples', default=10132, help='number of examples for evaluation', type=int)
parser.add_argument('--min_global_step', default=5000, help='minimum global step to run evaluation', type=int)
parser.add_argument('--dataset_dir', default='meme_characters/', help='directory where memes are stored')
parser.add_argument('--vocab_file', default='meme_characters/word_count.txt', help='vocabulary file')
parser.add_argument('--model_file', default='inception_log3.0/fine_inception.h5', help='model file')
args = parser.parse_args()

def evaluate_model(sess, model, global_step, summary_writer, summary_op):
    """Computes perplexity-per-word over the evaluation dataset.
    Summaries and perplexity-per-word are written out to the eval directory.
    Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
    """
    # Log model summaries on a single batch.
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, global_step)

    # Compute perplexity over the entire dataset.
    num_eval_batches = int(
        math.ceil(args.num_eval_examples / config.batch_size))

    start_time = time.time()
    sum_losses = 0.
    sum_weights = 0.
    for i in range(num_eval_batches):
        cross_entropy_losses, weights = sess.run([
            model.target_cross_entropy_losses,
            model.target_cross_entropy_loss_weights
        ])
        sum_losses += np.sum(cross_entropy_losses * weights)
        sum_weights += np.sum(weights)
        if not i % 100:
            tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                            num_eval_batches)
    eval_time = time.time() - start_time

    perplexity = math.exp(sum_losses / sum_weights)
    tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)

    # Log perplexity to the FileWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = perplexity
    value.tag = "Perplexity"
    summary_writer.add_summary(summary, global_step)

    # Write the Events file to the eval directory.
    summary_writer.flush()
    tf.logging.info("Finished processing evaluation at global step %d.",
                    global_step)


def run_once(model, saver, summary_writer, summary_op):
    """Evaluates the latest model checkpoint.
    Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
    """
    model_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    if not model_path:
        tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                        args.checkpoint_dir)
        return

    with tf.Session() as sess:
        # Load model from checkpoint.
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step.name)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_path), global_step)
        if global_step < args.min_global_step:
            tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                            args.min_global_step)
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Run evaluation on the latest checkpoint.
        try:
            evaluate_model(
                sess=sess,
                model=model,
                global_step=global_step,
                summary_writer=summary_writer,
                summary_op=summary_op)
        except Exception as e:  # pylint: disable=broad-except
            tf.logging.error("Evaluation failed.")
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def run():
    """Runs evaluation in a loop, and logs summaries to TensorBoard."""
    # Create the evaluation directory if it doesn't exist.
    eval_dir = args.eval_dir
    if not tf.gfile.IsDirectory(eval_dir):
        tf.logging.info("Creating eval directory: %s", eval_dir)
        tf.gfile.MakeDirs(eval_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation.
        model = MemeModel('eval', args.vocab_file,
                          dataset_dir=args.dataset_dir,
                          model_file=args.model_file)
        model.build()

        # Create the Saver to restore model Variables.
        saver = tf.train.Saver()

        # Create the summary operation and the summary writer.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir)

        g.finalize()

        # Run a new evaluation run every eval_interval_secs.
        while True:
            start = time.time()
            tf.logging.info("Starting evaluation at " + time.strftime(
                "%Y-%m-%d-%H:%M:%S", time.localtime()))
            run_once(model, saver, summary_writer, summary_op)
            time_to_next_eval = start + args.eval_interval_secs - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)


if __name__ == "__main__":
    run()
