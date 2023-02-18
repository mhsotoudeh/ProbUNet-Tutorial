rsync -a --progress --exclude "model*" --exclude "loss*" -e "ssh -i ~/.ssh/computecanada"                                                                             \
"sotoudeh@beluga.computecanada.ca:/home/sotoudeh/projects/rrg-lplevass/sotoudeh/ProbUNet-Tutorial/runs/"    \
"/mnt/d/Academia/Learning/Mini Projects/2023 Probabilistic U-Net Tutorial/ProbUNet-Tutorial/runs/"