import os
import glob
import pandas as pd
from datetime import datetime

def load_all_games(data_folder="./data"):
    """โหลดข้อมูลเกมทั้งหมดจาก Data Folder"""
    if not os.path.exists(data_folder):
        return []
    
    all_files = glob.glob(os.path.join(data_folder, "*.txt"))
    all_games = []
    
    for f in all_files:
        try:
            with open(f, 'r') as file:
                content = file.read().strip()
                raw = content.split(']')[1] if ']' in content else content
                seq = [int(x) for x in raw.replace(',', ' ').split() if x.strip().isdigit()]
                all_games.append([x for x in seq if x != 2]) # Filter ties
        except Exception as e:
            print(f"Error loading game {f}: {e}")
    return all_games

def load_custom_patterns(folder_path="./pattern"):
    """โหลด Pattern จากโฟลเดอร์ pattern/"""
    patterns = []
    
    if not os.path.exists(folder_path):
        return patterns
    
    pattern_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for filepath in pattern_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if '=' in line:
                        pattern_str, expected_str = line.split('=')
                        pattern = [int(x) for x in pattern_str.split(',')]
                        expected = int(expected_str)
                    else:
                        pattern = [int(x) for x in line.split(',')]
                        expected = None
                        
                    patterns.append({
                        'pattern': pattern,
                        'expected': expected,
                        'name': os.path.basename(filepath).replace('.txt', '')
                    })
        except Exception as e:
            print(f"Error loading pattern {filepath}: {e}")
    
    return patterns

def calculate_streak(history):
    """คำนวณความยาว streak ปัจจุบัน"""
    if not history:
        return 0
    non_tie = [h for h in history if h != 2]
    if not non_tie:
        return 0
    streak = 1
    last_winner = non_tie[-1]
    for i in range(len(non_tie) - 2, -1, -1):
        if non_tie[i] == last_winner:
            streak += 1
        else:
            break
    return streak

def calculate_tie_stats(history, window=20):
    """คำนวณสถิติ Tie"""
    if not history:
        return 0, 0
    last_n = history[-window:]
    tie_rate = last_n.count(2) / len(last_n) if last_n else 0
    gap = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i] == 2:
            break
        gap += 1
    return tie_rate, gap

def save_game_data(history, data_folder="./data"):
    """บันทึกข้อมูลเกมลงไฟล์ .txt"""
    if not history:
        return None
    
    os.makedirs(data_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_{timestamp}.txt"
    filepath = os.path.join(data_folder, filename)
    
    data_str = ", ".join(str(x) for x in history)
    with open(filepath, 'w') as f:
        f.write(data_str)
    
    return filename
