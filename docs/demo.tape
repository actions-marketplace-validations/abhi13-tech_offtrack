# vhs script for the README demo GIF — reproducible, offline, $0.
# Regenerate: cd examples/refund-agent && vhs ../../docs/demo.tape

Output ../../docs/demo.gif

Set FontSize 15
Set Width 900
Set Height 620
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Hide
Type "rm -rf .offtrack && clear"
Enter
Show

Type "# 1. record golden trajectories with your current model"
Enter
Sleep 800ms
Type "AGENT_MODEL=fake-careful offtrack record"
Enter
Sleep 5s

Type "clear"
Enter
Type "# 2. swap the model (careful -> sloppy)... then check"
Enter
Sleep 1s
Type "AGENT_MODEL=fake-sloppy offtrack check"
Enter
Sleep 9s

Type "echo exit code: $?"
Enter
Sleep 4s
