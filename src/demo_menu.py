"""
demo_menu.py — Zero-typing launcher for the live project demonstration
======================================================================
Double-click RUN_DEMO.bat (or run: python src/demo_menu.py), then press a
number. Built for presenting to a guide/professor: every option is one
keypress, and the console prints a one-line reminder of what to say.

Controls inside every demo window:
    SPACE pause | R new scenario | +/- speed | ESC quit the window
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MENU = r"""
==========================================================================
   MARL COOPERATIVE PAYLOAD TRANSPORT - LIVE DEMO
   (AI-ML Semester Project - Guide: Dr. S. S. Ohol, COEP)
==========================================================================

  [1]  UNTRAINED agents (random) .............. the problem, unsolved
  [2]  CLASSICAL controller (APF) ............. the hand-engineered baseline
  [3]  QMIX - learned MARL team ............... THE MAIN RESULT
  [4]  MAPPO - learned MARL team .............. the careful/safe learner
  [5]  Open PROJECT PPT (the approved deck)
  [6]  Open REPORT (PDF)
  [7]  Re-run 30-episode benchmark (takes ~3 min, prints table)
  [8]  Open the plots folder (figures)

  Controls in every simulation window:
     SPACE  pause      R  new scenario      +/-  speed      ESC  quit

  Tip: press 1 first (show the difficulty), then 3 (show your solution).
==========================================================================
"""

TALKING = {
    "1": "Say: 'Random policy - the rovers achieve nothing. This is the hard "
         "part of the task.'",
    "2": "Say: 'Classical potential-fields controller: it works (100% success) "
         "BUT it uses exact ground-truth sensing. Watch it fail and back off "
         "when stuck.'",
    "3": "Say: 'Our trained MARL team - each rover decides using ONLY its own "
         "9-beam sensor. 73% success, 6x faster than MAPPO. This policy was "
         "LEARNED, not programmed.'",
    "4": "Say: 'MAPPO - slower but perfectly safe: ZERO collisions on every "
         "successful delivery. Different algorithm, same reward - different "
         "character.'",
    "7": "Say: 'The formal 30-episode evaluation - same seeds for every "
         "policy, so the comparison is fair.'",
}


def open_file(path: str):
    if os.path.exists(path):
        os.startfile(path)          # Windows default app
    else:
        print(f"  !! missing: {path}")


def run_benchmark():
    print("\nRunning 30-episode formal benchmark (QMIX)...")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "evaluate.py"),
                    "--ckpt", os.path.join(ROOT, "runs", "qmix_n3_s0",
                                           "ckpt", "final.pt"),
                    "--episodes", "30", "--out-name", "eval_qmix_final"],
                   cwd=ROOT)


def main():
    while True:
        print(MENU)
        choice = input("Select option (number, or q to quit): ").strip()
        if choice in ("q", "quit", ""):
            break
        if choice in TALKING:
            print("\n>> " + TALKING[choice] + "\n")
        try:
            if choice == "1":
                subprocess.run([sys.executable, os.path.join(ROOT, "src",
                                "demo.py"), "--random"], cwd=ROOT)
            elif choice == "2":
                subprocess.run([sys.executable, os.path.join(ROOT, "src",
                                "demo.py"), "--baseline"], cwd=ROOT)
            elif choice == "3":
                subprocess.run([sys.executable, os.path.join(ROOT, "src",
                                "demo.py"), "--ckpt", os.path.join(
                                    ROOT, "runs", "qmix_n3_s0", "ckpt",
                                    "final.pt")], cwd=ROOT)
            elif choice == "4":
                subprocess.run([sys.executable, os.path.join(ROOT, "src",
                                "demo.py"), "--ckpt", os.path.join(
                                    ROOT, "runs", "mappo_n3_cur", "ckpt",
                                    "best.pt")], cwd=ROOT)
            elif choice == "5":
                # The approved project PPT (copied into the project so the
                # demo is self-contained; falls back to the Documents copy).
                ppt_local = os.path.join(ROOT, "topic of project.pptx")
                if os.path.exists(ppt_local):
                    open_file(ppt_local)
                else:
                    ppt_docs = os.path.join(os.path.expanduser("~"),
                                            "Documents",
                                            "topic of project.pptx")
                    if os.path.exists(ppt_docs):
                        open_file(ppt_docs)
                    else:
                        print("  !! project PPT not found")
            elif choice == "6":
                open_file(os.path.join(ROOT, "REPORT.pdf"))
            elif choice == "7":
                run_benchmark()
            elif choice == "8":
                open_file(os.path.join(ROOT, "plots"))
            else:
                print("  (unknown option)")
        except KeyboardInterrupt:
            pass
        input("\n[press ENTER to return to the menu]")


if __name__ == "__main__":
    main()
