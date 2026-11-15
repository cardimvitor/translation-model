#!/bin/sh
# v12 (restart-safe): post-train v11 on mixed-language pairs.
cd "$(dirname "$0")/.."; W=build/work
if [ ! -f $W/v12_post.done ]; then
  sh build/post_train.sh student_V11 student_V12 ${V12_MIN:-120} pairs_NEW_v12.jsonl >> $W/v12_post.log 2>&1; echo "post_train exit $?" >> $W/v12_post.log; touch $W/v12_post.done
fi
