Okay so guys here’s a serious set of steps to follow every single time you want to do anything not local:

1. Switch to your local main branch using vsc/terminal
2. Do a git pull
3. Swap back to your local name branch (mine would be tanishq)
4. Do a git fetch origin
5. Then do a git merge origin/main into your current branch

This makes sure that you have all the changes that anyone who merged before you made, and actually brings them into your local name branch to help prevent merge conflicts.

1. Do a git add .
2. git commit -m "comment"
3. git push (or wtv vsc/terminal suggests after you try git push)

NOW YOU MESSAGE ON THE GROUP saying you pushed to your branch

Next go to GitHub and create a PR to merge your name branch into main

NOW YOU MSG ON THE GRP TO ASK FOR A PR REVIEW
NEXT AFTER APPROVED AND DONE, YOU MERGE, AND SAY: guys I’ve merged please pull