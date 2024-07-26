#! /usr/bin/bash

export WANDB_USERNAME=apfurman

cd /opt/isaacgym/rsl_rl/
git pull

cd /opt/isaacgym/legged_gym/
git pull
git checkout $RUN_BRANCH

# get latest commit message, replace spaces with dashes, and prepend "b1-"
latest_commit_message=$(git log --no-merges -1 --format=%s)
exptid="b1-${latest_commit_message// /-}"

# train
python legged_gym/scripts/train.py --task=b1 --exptid=$exptid --headless

# get video of training
python legged_gym/scripts/play.py --task=b1 --exptid=$exptid --headless

cd /opt/isaacgym/output_files/dog_walk/$exptid/
ffmpeg -f image2 -framerate 20 -i frames/%d.png -c:v libx264 -crf 22 export/video.mp4

echo "Done training! Use this command to copy exported files:"
echo "kubectl cp walk-anthony:/opt/isaacgym/output_files/dog_walk/$exptid/export/ ./runs/$exptid/"
echo