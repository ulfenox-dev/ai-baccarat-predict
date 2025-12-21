import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# ส่วนที่ 1: Backend Logic - Hybrid Ensemble AI
# ==========================================

def build_big_road_columns(history):
    """สร้างโครงสร้างคอลัมน์ของ Big Road"""
    columns = []
    current_col = []
    prev_winner = None
    
    for result in history:
        if result == 2:
            continue
        if prev_winner is None:
            current_col = [result]
            prev_winner = result
        elif result == prev_winner:
            current_col.append(result)
        else:
            columns.append(current_col)
            current_col = [result]
            prev_winner = result
    
    if current_col:
        columns.append(current_col)
    return columns

def get_derived_road_result(columns, road_type='big_eye'):
    """คำนวณเค้าไพ่รอง (Derived Roads)"""
    offset_map = {'big_eye': 1, 'small': 2, 'cockroach': 3}
    offset = offset_map.get(road_type, 1)
    derived_results = []
    
    if len(columns) < offset + 1:
        return derived_results
    
    for i in range(offset + 1, len(columns) + 1):
        current_cols = columns[:i]
        if len(current_cols) < offset + 1:
            continue
        current_col = current_cols[-1]
        compare_col = current_cols[-(offset + 1)]
        
        for row_idx in range(len(current_col)):
            if row_idx == 0:
                if len(compare_col) == len(current_cols[-2]) if len(current_cols) > 1 else True:
                    derived_results.append(1)
                else:
                    derived_results.append(0)
            else:
                if row_idx < len(compare_col):
                    derived_results.append(1)
                else:
                    derived_results.append(0)
    
    return derived_results

def get_derived_roads_features(history):
    """สร้าง Features จากเค้าไพ่รองทั้ง 3 แบบ (Updated to return raw roads)"""
    columns = build_big_road_columns(history)
    
    big_eye = get_derived_road_result(columns, 'big_eye')
    small_road = get_derived_road_result(columns, 'small')
    cockroach = get_derived_road_result(columns, 'cockroach')
    
    # คำนวณ features: สัดส่วน Red ใน 5 ตาหลังสุดของแต่ละ road
    def get_red_ratio(road, n=5):
        if not road:
            return 0.5
        last_n = road[-n:]
        return sum(last_n) / len(last_n) if last_n else 0.5
    
    big_eye_stability = get_red_ratio(big_eye)
    small_road_stability = get_red_ratio(small_road)
    cockroach_stability = get_red_ratio(cockroach)
    
    overall_stability = (big_eye_stability + small_road_stability + cockroach_stability) / 3
    
    return {
        'big_eye_stability': big_eye_stability,
        'small_road_stability': small_road_stability,
        'cockroach_stability': cockroach_stability,
        'overall_stability': overall_stability,
        'is_stable': 1 if overall_stability > 0.6 else 0,
        'is_volatile': 1 if overall_stability < 0.4 else 0,
        # Raw sequences for new Pattern Analysis
        'big_eye_seq': big_eye,
        'small_road_seq': small_road,
        'cockroach_seq': cockroach
    }

def analyze_road_quality(road_sequence):
    """
    วิเคราะห์คุณภาพของเค้าไพ่ (Pattern Analysis)
    Input: list of 0/1 (Derived Road results)
    Returns: score (int) - ยิ่งเยอะยิ่งน่าตาม
    """
    if not road_sequence or len(road_sequence) < 4:
        return 0
    
    score = 0
    last_n = road_sequence[-6:] # Analyze last 6 hands
    
    # 1. Dragon Pattern (Ending with same color)
    streak = 0
    if len(last_n) > 0:
        last_val = last_n[-1]
        for x in reversed(last_n):
            if x == last_val:
                streak += 1
            else:
                break
    
    if streak >= 4:
        score += streak * 2  # Strong pattern
    elif streak == 3:
        score += 2 # Potential dragon
        
    # 2. Ping Pong Pattern (Alternating)
    pingpong = 0
    if len(last_n) >= 4:
        is_pp = True
        for i in range(len(last_n) - 1, 0, -1):
            if last_n[i] == last_n[i-1]:
                is_pp = False
                break
        if is_pp:
            score += len(last_n) * 2
            
    return score

def simulate_next_move(history, next_result):
    """จำลองว่าถ้าตาถัดไปออก next_result แล้ว road จะเป็นยังไง"""
    simulated = history + [next_result]
    return get_derived_roads_features(simulated)

def get_technician_vote(history):
    """
    Module B: The Technician (Refactored)
    Vote based on Pattern Continuation (Dragon/Ping Pong) in Derived Roads
    """
    if len([h for h in history if h != 2]) < 5:
        return None, 0
    
    # Simulate both outcomes
    player_sim = simulate_next_move(history, 0)
    banker_sim = simulate_next_move(history, 1)
    
    # Analyze Big Eye patterns (Primary focus)
    p_score = analyze_road_quality(player_sim['big_eye_seq'])
    b_score = analyze_road_quality(banker_sim['big_eye_seq'])
    
    # Add weights from Small Road & Cockroach
    p_score += analyze_road_quality(player_sim['small_road_seq']) * 0.5
    b_score += analyze_road_quality(banker_sim['small_road_seq']) * 0.5
    
    p_score += analyze_road_quality(player_sim['cockroach_seq']) * 0.5
    b_score += analyze_road_quality(banker_sim['cockroach_seq']) * 0.5
    
    # Normalize score for display (max usually around 10-15)
    confidence_display = max(p_score, b_score)
    
    if p_score > b_score + 2: # Significant difference required
        return 0, int(confidence_display)
    elif b_score > p_score + 2:
        return 1, int(confidence_display)
    else:
        return None, 0

# ==========================================
# Module D: Expert Rules (Custom Patterns)
# ==========================================

def load_all_games(data_folder="./data"):
    """โหลดข้อมูลเกมทั้งหมดจาก Data Folder"""
    all_files = glob.glob(os.path.join(data_folder, "*.txt"))
    all_games = []
    
    if not os.path.exists(data_folder):
        return []
        
    for f in all_files:
        try:
            content = open(f, 'r').read().strip()
            raw = content.split(']')[1] if ']' in content else content
            seq = [int(x) for x in raw.replace(',', ' ').split() if x.strip().isdigit()]
            all_games.append([x for x in seq if x != 2]) # Filter ties
        except: pass
    return all_games

def load_custom_patterns(folder_path="./pattern"):
    """
    โหลด Pattern จากโฟลเดอร์ pattern/
    Supports both:
    1. '0,1,0,1=0' (Fixed expectation)
    2. '0,1,0,1'   (Auto calculation needed)
    """
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
                        # No expected value = to be calculated from data
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

def get_expert_vote(history, patterns):
    """
    Module D: Expert Rules
    Returns: (vote, matched_pattern_name)
    Only considers patterns that have a valid 'expected' value (0 or 1).
    """
    if not patterns or not history:
        return None, None
    
    # Filter out ties for pattern matching
    non_tie = [h for h in history if h != 2]
    
    for p in patterns:
        if p['expected'] is None:
            continue
            
        pattern = p['pattern']
        pattern_len = len(pattern)
        
        if len(non_tie) >= pattern_len:
            # Check if last N hands match the pattern
            last_n = non_tie[-pattern_len:]
            if last_n == pattern:
                return p['expected'], p['name']
    
    return None, None

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

def process_data_from_folder(folder_path):
    """อ่านไฟล์ .txt ทั้งหมดใน data folder"""
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))
    data_rows = []
    pattern_sequences = []  # For KNN
    
    if not all_files:
        return pd.DataFrame(), []

    for filepath in all_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                raw = content.split(']')[1] if ']' in content else content
                game_seq = [int(x) for x in raw.replace(',', ' ').split() if x.strip().isdigit()]
                
                for i in range(5, len(game_seq)):
                    history = game_seq[:i]
                    target = game_seq[i]
                    
                    non_tie_history = [h for h in history if h != 2]
                    if len(non_tie_history) < 5:
                        continue
                    
                    # Features for RF
                    p1, p2, p3, p4, p5 = non_tie_history[-5:]
                    derived_features = get_derived_roads_features(history)
                    current_streak = calculate_streak(history)
                    tie_rate, gap_since_tie = calculate_tie_stats(history, 20)
                    last_10 = [h for h in history[-10:] if h != 2]
                    b_ratio = last_10.count(1) / len(last_10) if last_10 else 0.5
                    
                    row = {
                        'pattern_1': p1, 'pattern_2': p2, 'pattern_3': p3,
                        'pattern_4': p4, 'pattern_5': p5,
                        'current_streak': min(current_streak, 10),
                        'banker_trend': b_ratio,
                        'tie_rate_20': tie_rate,
                        'gap_since_tie': min(gap_since_tie, 30),
                        'is_stable': derived_features['is_stable'],
                        'overall_stability': derived_features['overall_stability'],
                        'target': target
                    }
                    data_rows.append(row)
                    
                    # Pattern sequence for KNN (last 5 results as features)
                    pattern_sequences.append({
                        'pattern': non_tie_history[-5:],
                        'target': target if target != 2 else non_tie_history[-1]  # Skip ties for KNN
                    })
                    
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return pd.DataFrame(data_rows), pattern_sequences

def train_ensemble_models(df, pattern_sequences):
    """Train both RF and KNN models"""
    models = {}
    
    if not df.empty:
        # Random Forest
        X = df.drop(columns=['target'])
        y = df['target']
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
        rf_model.fit(X, y)
        models['rf'] = rf_model
        models['rf_features'] = list(X.columns)
        
        # KNN for pattern matching
        if pattern_sequences:
            knn_X = np.array([p['pattern'] for p in pattern_sequences])
            knn_y = np.array([p['target'] for p in pattern_sequences])
            # Filter out ties from y
            mask = knn_y != 2
            knn_X = knn_X[mask]
            knn_y = knn_y[mask]
            
            if len(knn_X) > 10:
                knn_model = KNeighborsClassifier(n_neighbors=min(7, len(knn_X)//3), weights='distance')
                knn_model.fit(knn_X, knn_y)
                models['knn'] = knn_model
    
    return models

def ensemble_predict(history, models, module_performance=None):
    """
    Hybrid Ensemble Voting System
    Returns: prediction, score, vote_details
    """
    votes = {'player': 0, 'banker': 0}
    vote_details = {}
    
    non_tie_hist = [h for h in history if h != 2]
    if len(non_tie_hist) < 5:
        return None, 0, {"error": "Need more data"}
    
    # ========== Module A: KNN Historian (2 points) ==========
    if 'knn' in models:
        pattern = np.array(non_tie_hist[-5:]).reshape(1, -1)
        knn_pred = models['knn'].predict(pattern)[0]
        knn_proba = models['knn'].predict_proba(pattern)[0]
        knn_conf = max(knn_proba) * 100
        
        if knn_pred == 0:
            votes['player'] += 2
            vote_details['historian'] = {'vote': 'PLAYER', 'conf': knn_conf, 'emoji': '📜'}
        else:
            votes['banker'] += 2
            vote_details['historian'] = {'vote': 'BANKER', 'conf': knn_conf, 'emoji': '📜'}
    else:
        vote_details['historian'] = {'vote': 'N/A', 'conf': 0, 'emoji': '📜'}
    
    # ========== Module B: Technician - Derived Roads (1 point) ==========
    tech_vote, roads_agree = get_technician_vote(history)
    if tech_vote is not None:
        if tech_vote == 0:
            votes['player'] += 1
            vote_details['technician'] = {'vote': 'PLAYER', 'roads': roads_agree, 'emoji': '🛣️'}
        else:
            votes['banker'] += 1
            vote_details['technician'] = {'vote': 'BANKER', 'roads': roads_agree, 'emoji': '🛣️'}
    else:
        vote_details['technician'] = {'vote': 'NEUTRAL', 'roads': 0, 'emoji': '🛣️'}
    
    # ========== Module C: RF Statistician (1 point if >55%) ==========
    if 'rf' in models:
        p1, p2, p3, p4, p5 = non_tie_hist[-5:]
        derived_features = get_derived_roads_features(history)
        current_streak = calculate_streak(history)
        tie_rate, gap_since_tie = calculate_tie_stats(history, 20)
        last_10 = [h for h in history[-10:] if h != 2]
        b_ratio = last_10.count(1) / len(last_10) if last_10 else 0.5
        
        input_data = pd.DataFrame([{
            'pattern_1': p1, 'pattern_2': p2, 'pattern_3': p3,
            'pattern_4': p4, 'pattern_5': p5,
            'current_streak': min(current_streak, 10),
            'banker_trend': b_ratio,
            'tie_rate_20': tie_rate,
            'gap_since_tie': min(gap_since_tie, 30),
            'is_stable': derived_features['is_stable'],
            'overall_stability': derived_features['overall_stability']
        }])
        
        rf_pred = models['rf'].predict(input_data)[0]
        rf_proba = models['rf'].predict_proba(input_data)[0]
        rf_conf = max(rf_proba) * 100
        
        if rf_conf > 55:  # Only count if confident enough
            if rf_pred == 0:
                votes['player'] += 1
            elif rf_pred == 1:
                votes['banker'] += 1
        
        vote_text = 'PLAYER' if rf_pred == 0 else ('BANKER' if rf_pred == 1 else 'TIE')
        vote_details['statistician'] = {'vote': vote_text, 'conf': rf_conf, 'emoji': '🧠'}
    else:
        vote_details['statistician'] = {'vote': 'N/A', 'conf': 0, 'emoji': '🧠'}
    
    # ========== Module D: Expert Rules (2 points - highest weight!) ==========
    if 'patterns' in models and models['patterns']:
        expert_vote, pattern_name = get_expert_vote(history, models['patterns'])
        if expert_vote is not None:
            if expert_vote == 0:
                votes['player'] += 2
                vote_details['expert'] = {'vote': 'PLAYER', 'pattern': pattern_name, 'emoji': '🎲'}
            else:
                votes['banker'] += 2
                vote_details['expert'] = {'vote': 'BANKER', 'pattern': pattern_name, 'emoji': '🎲'}
        else:
            vote_details['expert'] = {'vote': 'NO MATCH', 'pattern': None, 'emoji': '🎲'}
    else:
        vote_details['expert'] = {'vote': 'N/A', 'pattern': None, 'emoji': '🎲'}
        
    # ========== Session Learning (Adaptive Feedback - Trap Avoidance) ==========
    session_penalty = 0
    learned_msg = None
    
    if 'session_mistakes' in st.session_state and st.session_state['session_mistakes']:
        current_pat = [h for h in history[-5:] if h != 2]
        for mistake in st.session_state['session_mistakes']:
            if len(current_pat) >= 3 and current_pat == mistake['pattern']:
                if mistake['wrong_predict'] == 0:
                    votes['player'] -= 2
                    session_penalty = -2
                    learned_msg = "⚠️ เคยผิด Pattern นี้ (ลด Player)"
                elif mistake['wrong_predict'] == 1:
                    votes['banker'] -= 2
                    session_penalty = -2
                    learned_msg = "⚠️ เคยผิด Pattern นี้ (ลด Banker)"
    
    if learned_msg:
        vote_details['learning'] = {'msg': learned_msg, 'penalty': session_penalty}

    # ========== Dynamic Weighting (Continuous Thinking) ==========
    if module_performance:
        for mod, perf_score in module_performance.items():
            if mod in vote_details:
                # Get module vote
                mod_vote = vote_details[mod].get('vote')
                bonus = 0
                
                # If Hot (Score >= 2) -> Add Weight +0.5
                if perf_score >= 2:
                    bonus = 0.5
                # If Cold (Score <= -2) -> Reduce Weight -0.5 (or penalty)
                elif perf_score <= -2:
                    bonus = -0.5
                
                if bonus != 0 and mod_vote in ['PLAYER', 'BANKER']:
                    target = 'player' if mod_vote == 'PLAYER' else 'banker'
                    votes[target] += bonus
                    vote_details[mod]['dynamic_bonus'] = bonus

    # ========== Council Decision ==========
    total_score = max(votes['player'], votes['banker'])
    final_prediction = 0 if votes['player'] > votes['banker'] else 1
    
    return final_prediction, total_score, vote_details

# ==========================================
# ส่วนที่ 2: หน้าจอใช้งาน (GUI)
# ==========================================

st.set_page_config(page_title="AI Baccarat Pro", layout="wide")

st.title("🤖 AI Baccarat Pro (เวอร์ชั่นจับสูตรหลอก)")
st.markdown("ระบบวิเคราะห์ไพ่ 4 มิติ: **กราฟ + เทคนิค + สถิติ + กับดัก**")

# --- เมนูด้านซ้าย: จัดการข้อมูล ---
with st.sidebar:
    st.header("📂 1. เตรียมสมอง AI")
    data_path = st.text_input("โฟลเดอร์เก็บไฟล์ .txt", value="./data")
    
    # Initialize Session Learning State
    if 'session_mistakes' not in st.session_state:
        st.session_state['session_mistakes'] = [] # List of {'pattern': [], 'wrong_vote': 0/1}
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
    if 'module_performance' not in st.session_state:
        # Track score: +1 if correct, -1 if wrong
        st.session_state['module_performance'] = {
            'historian': 0, 'technician': 0, 'statistician': 0, 'expert': 0
        }
    if 'last_vote_details' not in st.session_state:
        st.session_state['last_vote_details'] = {}
    if 'ai_performance' not in st.session_state:
        st.session_state['ai_performance'] = {'wins': 0, 'losses': 0, 'history': []}
    if 'undo_stack' not in st.session_state:
        st.session_state['undo_stack'] = [] # List of delta dicts
    
    if st.button("🔄 โหลดข้อมูล & เทรน Ensemble", use_container_width=True):
        with st.spinner("กำลังเทรน AI 4 ตัว..."):
            df, pattern_sequences = process_data_from_folder(data_path)
            
            if not df.empty:
                models = train_ensemble_models(df, pattern_sequences)
                
                # Load custom patterns & Auto-calculate expectation if needed
                custom_patterns = load_custom_patterns("./pattern")
                
                # Check if any pattern needs calculation
                if any(p['expected'] is None for p in custom_patterns):
                    all_games = load_all_games(data_path)
                    
                    for p in custom_patterns:
                        if p['expected'] is None:
                            # Calculate stats
                            pat_seq = p['pattern']
                            pat_len = len(pat_seq)
                            next_p = 0
                            next_b = 0
                            
                            for game in all_games:
                                for i in range(len(game) - pat_len):
                                    if game[i:i+pat_len] == pat_seq:
                                        if i+pat_len < len(game):
                                            nxt = game[i+pat_len]
                                            if nxt == 0: next_p += 1
                                            elif nxt == 1: next_b += 1
                            
                            total = next_p + next_b
                            if total > 0:
                                p_rate = next_p / total
                                b_rate = next_b / total
                                
                                if p_rate >= 0.55:
                                    p['expected'] = 0 # Vote Player
                                elif b_rate >= 0.55:
                                    p['expected'] = 1 # Vote Banker
                                else:
                                    p['expected'] = None # No clear vote
                
                models['patterns'] = custom_patterns
                
                st.session_state['models'] = models
                st.session_state['data_count'] = len(df)
                
                # Summary
                model_count = sum([1 for k in ['rf', 'knn'] if k in models])
                pattern_count = len(custom_patterns)
                st.success(f"✅ เทรนเสร็จ! {model_count} AI + {pattern_count} Patterns")
            else:
                st.error("ไม่เจอไฟล์ข้อมูลในโฟลเดอร์ data")
                
    st.divider()
    
    # Function to render big road (Moved to global scope)
    def render_big_road(history, mini=False):
        ROWS = 6
        # User requested 12 columns for the grid and larger size
        COLS = 30 if not mini else 12 
        grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        

        p_count = history.count(0)
        b_count = history.count(1)
        t_count = history.count(2)
        total = len(history)
        
        col = 0
        row = 0
        prev_winner = None
        
        for idx, item in enumerate(history):
            try:
                winner = int(item)
            except:
                continue
            if winner == 2:
                if prev_winner is None:
                    grid[0][0] = {'color': 'green', 'tie_count': 1}
                elif row < ROWS and col < COLS and grid[row][col]:
                    grid[row][col]['tie_count'] = grid[row][col].get('tie_count', 0) + 1
                continue
            color = 'blue' if winner == 0 else 'red'
            if prev_winner is None:
                grid[row][col] = {'color': color}
                prev_winner = winner
            elif winner == prev_winner:
                next_row = row + 1
                if next_row < ROWS and grid[next_row][col] is None:
                    row = next_row
                    grid[row][col] = {'color': color}
                else:
                    col += 1
                    if col < COLS: grid[row][col] = {'color': color}
            else:
                col += 1
                row = 0
                if col < COLS: grid[row][col] = {'color': color}
                prev_winner = winner

        # Build HTML - NO INDENTATION in the string
        cell_size = "28px" if not mini else "18px"
        circle_size = "20px" if not mini else "14px"
        font_size = "14px" if not mini else "11px"
        
        # Theme: White Background as requested
        # Borders need to be visible on white (use light grey)
        bg_color = "#ffffff" 
        border_color = "#e0e0e0"
        
        # UI Logic: Gradient for Main View, Transparent for Gallery Card
        container_bg = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" if not mini else "transparent"
        container_padding = "15px" if not mini else "0"
        container_radius = "12px" if not mini else "0"
        
        html = f"""
<style>
.baccarat-container {{ background: {container_bg}; padding: {container_padding}; border-radius: {container_radius}; margin-bottom: 15px; }}
.stats-bar {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 10px; border: 1px solid #333; }}
.stat-item {{ text-align: center; color: #ccc; font-size: {font_size}; }}
.stat-value {{ font-size: {20 if not mini else 12}px; font-weight: bold; color: #fff; }}
.stat-label {{ font-size: 10px; opacity: 0.6; }}
.grid-wrapper {{ overflow-x: auto; background: {bg_color}; border-radius: 4px; padding: 2px; border: 1px solid {border_color}; width: fit-content; margin: 0 auto; }}
.big-road-table {{ border-collapse: collapse; margin: 0 auto; }}
.big-road-table td {{ width: {cell_size}; height: {cell_size}; border: 1px solid {border_color}; padding: 0; position: relative; }}
.circle {{ width: {circle_size}; height: {circle_size}; border-radius: 50%; margin: auto; border-width: {3 if not mini else 2}px; border-style: solid; box-sizing: border-box; position: relative; background: transparent; }}
.circle.blue {{ border-color: #2196F3; }}
.circle.red {{ border-color: #f44336; }}
.circle.green {{ border-color: #4CAF50; background: transparent; }}
.tie-marker {{ position: absolute; top: -4px; right: -4px; background: #69F0AE; color: #000; font-size: 8px; width: 12px; height: 12px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; z-index:2; }}
</style>
<div class="baccarat-container">
{'<div class="stats-bar">' if not mini else ''}
{'<div class="stat-item"><div class="stat-label">HAND</div><div class="stat-value">#' + str(total) + '</div></div>' if not mini else ''}
{'<div class="stat-item"><div class="stat-label" style="color:#448AFF">P</div><div class="stat-value" style="color:#448AFF">' + str(p_count) + '</div></div>' if not mini else ''}
{'<div class="stat-item"><div class="stat-label" style="color:#FF5252">B</div><div class="stat-value" style="color:#FF5252">' + str(b_count) + '</div></div>' if not mini else ''}
{'<div class="stat-item"><div class="stat-label" style="color:#69F0AE">T</div><div class="stat-value" style="color:#69F0AE">' + str(t_count) + '</div></div>' if not mini else ''}
{'</div>' if not mini else ''}
<div class="grid-wrapper"><table class="big-road-table">
"""
        for r in range(ROWS):
            html += '<tr>'
            for c in range(COLS):
                cell = grid[r][c]
                if cell:
                    tie_html = f'<div class="tie-marker">{cell["tie_count"]}</div>' if cell.get('tie_count') else ''
                    html += f'<td><div class="circle {cell["color"]}">{tie_html}</div></td>'
                else:
                    html += '<td></td>'
            html += '</tr>'
        html += '</table></div></div>'
        return html

    # --- Pattern Validator Feature ---
    st.subheader("🔍 ตรวจสอบ Pattern")
    
    # --- Gallery Modal Helper ---
    # Attempt to use st.dialog (Streamlit 1.34+) or experimental_dialog
    gallery_decorator = getattr(st, "dialog", getattr(st, "experimental_dialog", None))
    
    if gallery_decorator:
        @gallery_decorator("🔍 Pattern Gallery", width="large")
        def show_gallery_modal(patterns):
            st.caption(f"📂 ทั้งหมด {len(patterns)} Patterns")
            
            # Grid Layout (4 columns to fit larger grids)
            cols = st.columns(4)
            for idx, p in enumerate(patterns):
                 with cols[idx % 4]:
                     # Use render_big_road with mini=True
                     grid_html = render_big_road(p['pattern'], mini=True)
                     
                     predict = "???"
                     if p.get('expected') == 0: predict = "<span style='color:#448AFF'>PLAYER</span>"
                     elif p.get('expected') == 1: predict = "<span style='color:#FF5252'>BANKER</span>"
                     
                     # Clean up filename (keep full name, just remove extension)
                     display_name = p['name'].replace('.txt', '')
                     
                     html_block = f"""
    <div style="margin-bottom:16px; border:1px solid #444; border-radius:12px; padding:15px; background:#1e1e24; box-shadow: 0 4px 15px rgba(0,0,0,0.4); display: flex; flex-direction: column; align-items: center;">
    <div style="width: 100%; display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding: 0 5px;">
        <div style="font-size:14px; color:#aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px;">📄 {display_name}</div>
        <div style="font-size:14px; font-weight:bold; color:#fff;">{predict}</div>
    </div>
    <div style="width: 100%; display: flex; justify-content: center;">
        {grid_html}
    </div>
    </div>
    """
                     st.markdown(html_block, unsafe_allow_html=True)
        
        @gallery_decorator("📊 Pattern Statistics", width="large")
        def show_stats_modal(patterns, data_path):
             st.info("กำลังวิเคราะห์ Pattern จาก Data ทั้งหมด...")
             
             # 1. Load data
             all_games = load_all_games(data_path)
             if not all_games:
                 st.error("ไม่พบข้อมูลเกม")
                 return

             results = []
             progress_bar = st.progress(0)
             
             for idx, p in enumerate(patterns):
                 pat_seq = p['pattern']
                 pat_len = len(pat_seq)
                 found = 0
                 next_p = 0
                 next_b = 0
                 
                 for game in all_games:
                     for i in range(len(game) - pat_len):
                         if game[i:i+pat_len] == pat_seq:
                             found += 1
                             if i+pat_len < len(game):
                                 nxt = game[i+pat_len]
                                 if nxt == 0: next_p += 1
                                 elif nxt == 1: next_b += 1
                 
                 # Calculate stats
                 total_valid = next_p + next_b
                 p_rate = (next_p / total_valid * 100) if total_valid > 0 else 0
                 b_rate = (next_b / total_valid * 100) if total_valid > 0 else 0
                 
                 rec = "⚪ 50/50"
                 if p_rate > 55: rec = "🔵 PLAYER"
                 elif b_rate > 55: rec = "🔴 BANKER"
                 
                 results.append({
                     "ชื่อไฟล์": p['name'],
                     "Pattern": str(pat_seq),
                     "เจอ (ครั้ง)": found,
                     "Next P%": f"{p_rate:.1f}%",
                     "Next B%": f"{b_rate:.1f}%",
                     "แนะนำ": rec
                 })
                 progress_bar.progress((idx + 1) / len(patterns))
                 
             st.dataframe(pd.DataFrame(results), use_container_width=True, height=500)
             st.success(f"วิเคราะห์เสร็จสิ้น {len(patterns)} Patterns")

    else:
        def show_gallery_modal(patterns): st.error("Update Streamlit")
        def show_stats_modal(patterns, path): st.error("Update Streamlit")

    col_pat1, col_pat2 = st.columns(2)
    check_btn = col_pat1.button("Check Stats", use_container_width=True)
    gallery_btn = col_pat2.button("See Gallery", use_container_width=True)
    
    if gallery_btn:
        patterns = load_custom_patterns("./pattern")
        if not patterns:
             st.warning("Empty folder")
        else:
             show_gallery_modal(patterns)
                 
    if check_btn:
        patterns = load_custom_patterns("./pattern")
        if not patterns:
             st.warning("ไม่เจอไฟล์ Pattern")
        else:
             show_stats_modal(patterns, data_path)

    st.divider()
    
    # Risk Level Slider
    st.subheader("⚡ Risk Level")
    risk_level = st.select_slider(
        "ระดับความเสี่ยง",
        options=["Low", "Medium", "High"],
        value="Medium",
        help="Low = แทงเฉพาะตอนมั่นใจมาก, High = แทงบ่อยขึ้น"
    )
    st.session_state['risk_level'] = risk_level
    
    # Threshold based on risk
    if risk_level == "Low":
        st.caption("🛡️ Score ≥ 4 ถึงแทง")
    elif risk_level == "Medium":
        st.caption("⚖️ Score ≥ 3 ถึงแทง")
    else:
        st.caption("🔥 Score ≥ 2 ถึงแทง")
        
    if 'session_mistakes' in st.session_state and st.session_state['session_mistakes']:
        st.caption(f"🧠 Adaptive Mode: เรียนรู้แล้ว {len(st.session_state['session_mistakes'])} จุด")
    
    st.divider()
    if 'models' in st.session_state:
        m = st.session_state['models']
        status_parts = []
        if 'knn' in m:
            status_parts.append("📜 KNN")
        if 'rf' in m:
            status_parts.append("🧠 RF")
        st.info(f"Active: {' + '.join(status_parts)}")
        st.caption(f"Data: {st.session_state['data_count']} samples")
    else:
        st.warning("⚠️ กรุณากดเทรนก่อน")


col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎰 2. บันทึกผลไพ่สด (Live Input)")
    

    if 'streak_count' not in st.session_state:
        st.session_state['streak_count'] = 0

    if 'ai_performance' in st.session_state:
        stats = st.session_state['ai_performance']
        wins = stats['wins']
        losses = stats['losses']
        total = wins + losses
        win_rate = (wins/total*100) if total > 0 else 0
        
        # Side Distribution Stats
        curr_game = st.session_state.get('current_game', [])
        p_pct, b_pct = 50, 50
        if curr_game:
            total_g = len(curr_game)
            p_pct = curr_game.count(0) / total_g * 100
            b_pct = curr_game.count(1) / total_g * 100
        
        # Brief Stats (Wins, Losses, Streak, Win Rate)
        st.markdown(f"""
        <div style="background:#2d2d44; padding:10px; border-radius:10px; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <div style="display:flex; justify-content:space-around; align-items:center; margin-bottom:10px; flex-wrap: wrap; gap: 5px;">
                <div style="color:#4CAF50; font-weight:bold; font-size:14px;">✅ Win: {wins}</div>
                <div style="color:#f44336; font-weight:bold; font-size:14px;">❌ Loss: {losses}</div>
                <div style="color:#FFC107; font-weight:bold; font-size:14px;">🔥 Streak: {st.session_state['streak_count']}</div>
                <div style="color:#00BCD4; font-weight:bold; font-size:14px;">📈 WinRate: {win_rate:.0f}%</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; opacity:0.8; background:rgba(0,0,0,0.2); padding:5px 10px; border-radius:4px;">
                <span style="color:#2196F3">PLAYER: {p_pct:.0f}%</span>
                <span style="color:#f44336">BANKER: {b_pct:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Stats (Expander)
        with st.expander("📊 ดูสถิติละเอียด (Profit/Loss)"):
             st.info(f"💰 Net Score: {wins - losses} หน่วย (สมมติเดินเงิน 1 หน่วยเสมอ)")
             st.write("Recent History:")
             # Show last 10 entries reversed
             for log in list(reversed(stats['history']))[:10]:
                 st.caption(log)
    

    if 'current_game' not in st.session_state:
        st.session_state['current_game'] = []
    
    # Toggle state for view mode
    if 'show_raw_data' not in st.session_state:
        st.session_state['show_raw_data'] = False


    def render_big_road(history):
        ROWS = 6
        COLS = 30
        grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        

        p_count = history.count(0)
        b_count = history.count(1)
        t_count = history.count(2)
        total = len(history)
        
        col = 0
        row = 0
        prev_winner = None
        tie_counts = {}  # Track tie counts per cell
        
        for idx, item in enumerate(history):
            try:
                winner = int(item)
            except:
                continue
            

            if winner == 2:
                if prev_winner is None:
                    grid[0][0] = {'color': 'green', 'tie_count': 1}
                elif row < ROWS and col < COLS and grid[row][col]:
                    grid[row][col]['tie_count'] = grid[row][col].get('tie_count', 0) + 1
                continue
            
            color = 'blue' if winner == 0 else 'red'
            
            if prev_winner is None:
                grid[row][col] = {'color': color}
                prev_winner = winner
            elif winner == prev_winner:
                next_row = row + 1
                if next_row < ROWS and grid[next_row][col] is None:
                    row = next_row
                    grid[row][col] = {'color': color}
                else:
                    col += 1
                    if col < COLS:
                        grid[row][col] = {'color': color}
            else:
                col += 1
                row = 0
                if col < COLS:
                    grid[row][col] = {'color': color}
                prev_winner = winner
        

        html = '''
        <style>
            .baccarat-container { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 12px; margin-bottom: 15px; }
            .stats-bar { display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; background: rgba(255,255,255,0.1); border-radius: 8px; margin-bottom: 10px; }
            .stat-item { text-align: center; color: #fff; font-size: 14px; }
            .stat-value { font-size: 20px; font-weight: bold; }
            .stat-label { font-size: 11px; opacity: 0.7; }
            .grid-wrapper { overflow-x: auto; background: #fff; border-radius: 8px; padding: 8px; }
            .big-road-table { border-collapse: collapse; }
            .big-road-table td { width: 28px; height: 28px; border: 1px solid #e0e0e0; padding: 0; position: relative; }
            .circle { width: 20px; height: 20px; border-radius: 50%; margin: auto; border-width: 3px; border-style: solid; box-sizing: border-box; position: relative; }
            .circle.blue { border-color: #2196F3; }
            .circle.red { border-color: #f44336; }
            .circle.green { border-color: #4CAF50; background: #4CAF50; }
            .tie-marker { position: absolute; top: -3px; right: -3px; background: #4CAF50; color: #fff; font-size: 9px; width: 14px; height: 14px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        </style>
        <div class="baccarat-container">
            <div class="stats-bar">
                <div class="stat-item"><div class="stat-value">''' + f'#{total}' + '''</div><div class="stat-label">HAND</div></div>
                <div class="stat-item"><div class="stat-value" style="color:#2196F3;">''' + f'{p_count}' + '''</div><div class="stat-label">PLAYER</div></div>
                <div class="stat-item"><div class="stat-value" style="color:#f44336;">''' + f'{b_count}' + '''</div><div class="stat-label">BANKER</div></div>
                <div class="stat-item"><div class="stat-value" style="color:#4CAF50;">''' + f'{t_count}' + '''</div><div class="stat-label">TIE</div></div>
            </div>
            <div class="grid-wrapper">
                <table class="big-road-table">
        '''
        
        for r in range(ROWS):
            html += '<tr>'
            for c in range(COLS):
                cell = grid[r][c]
                if cell:
                    clr = cell['color']
                    tie_count = cell.get('tie_count', 0)
                    tie_html = f'<span class="tie-marker">{tie_count}</span>' if tie_count > 0 else ''
                    html += f'<td><div class="circle {clr}">{tie_html}</div></td>'
                else:
                    html += '<td></td>'
            html += '</tr>'
        
        html += '</table></div></div>'
        return html


    def render_raw_data(history):
        if not history:
            return '<div style="background:#1a1a2e; color:#888; padding:20px; border-radius:12px; text-align:center; margin-bottom:15px;">ยังไม่มีข้อมูล</div>'
        
        mapping = {0: ('P', '#2196F3'), 1: ('B', '#f44336'), 2: ('T', '#4CAF50')}
        items_html = ''
        for i, val in enumerate(history):
            label, color = mapping.get(val, ('?', '#888'))
            items_html += f'<span style="display:inline-block; margin:3px; padding:5px 10px; background:{color}; color:#fff; border-radius:4px; font-weight:bold; font-size:12px;">{i+1}:{label}</span>'
        
        html = f'''
        <div style="background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding:15px; border-radius:12px; margin-bottom:15px;">
            <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px; max-height:200px; overflow-y:auto;">
                {items_html}
            </div>
            <div style="color:#888; font-size:11px; margin-top:8px; text-align:center;">รวม {len(history)} ตา | 0=P, 1=B, 2=T</div>
        </div>
        '''
        return html

    # Header with toggle button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("### 📊 ตาราง Big Road")
    with header_col2:
        if st.button("🔄 สลับมุมมอง", use_container_width=True):
            st.session_state['show_raw_data'] = not st.session_state['show_raw_data']
    
    # Render based on toggle state
    if st.session_state['show_raw_data']:
        st.caption("📝 มุมมอง: ข้อมูลดิบ (Raw Data)")
        st.markdown(render_raw_data(st.session_state['current_game']), unsafe_allow_html=True)
    else:
        st.caption("📊 มุมมอง: ตารางเค้าไพ่ (Grid)")
        st.markdown(render_big_road(st.session_state['current_game']), unsafe_allow_html=True)
    
    # Input buttons with gap
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    # Function to update Continuous Learning (Dynamic Weights)
    def update_learning(outcome):
        delta = {
            'added_mistake': None, # stores list index if added
            'module_changes': {},  # {'historian': +1, ...}
            'perf_change': None,   # {'type': 'win'/'loss', 'log': '...'}
            'prev_streak': st.session_state.get('streak_count', 0) # For Undo
        }
        
        # 1. Record Mistake for Session Penalty (-2 points) - Trap Avoidance
        if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
            last_pred = st.session_state['last_prediction']
            if last_pred['vote'] is not None and last_pred['vote'] != outcome and last_pred['score'] >= 3:
                history = st.session_state['current_game']
                non_tie = [h for h in history if h != 2]
                if len(non_tie) >= 5:
                    mistake_entry = {
                        'pattern': non_tie[-5:],
                        'wrong_predict': last_pred['vote']
                    }
                    st.session_state['session_mistakes'].append(mistake_entry)
                    delta['added_mistake'] = len(st.session_state['session_mistakes']) - 1
        
        # 2. Update Module Performance (Dynamic Weights) - Continuous Thinking
        if 'last_vote_details' in st.session_state and st.session_state['last_vote_details']:
            details = st.session_state['last_vote_details']
            perf = st.session_state['module_performance']
            
            def grade_module(name, module_data):
                if not module_data or 'vote' not in module_data: return
                vote_str = module_data['vote']
                if vote_str == 'PLAYER': vote_val = 0
                elif vote_str == 'BANKER': vote_val = 1
                else: return
                
                change = 1 if vote_val == outcome else -1
                perf[name] += change
                delta['module_changes'][name] = change
            
            grade_module('historian', details.get('historian'))
            grade_module('technician', details.get('technician'))
            grade_module('statistician', details.get('statistician'))
            grade_module('expert', details.get('expert'))
            
        # 3. Track Session Performance (Wins/Losses)
        if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
            last_pred = st.session_state['last_prediction']
            if last_pred['vote'] is not None and outcome != 2: # Ignore Ties checks
                is_win = (last_pred['vote'] == outcome)
                stats = st.session_state['ai_performance']
                
                req_score = 3
                curr_risk = st.session_state.get('risk_level', 'Medium')
                if curr_risk == 'Low': req_score = 4
                elif curr_risk == 'High': req_score = 2
                
                if last_pred['score'] >= req_score:
                    log_msg = ""
                    if is_win:
                        stats['wins'] += 1
                        log_msg = f"✅ Win ({last_pred['vote']} vs {outcome})"
                        stats['history'].append(log_msg)
                        delta['perf_change'] = {'type': 'win'}
                    else:
                        stats['losses'] += 1
                        log_msg = f"❌ Loss ({last_pred['vote']} vs {outcome})"
                        stats['history'].append(log_msg)
                        delta['perf_change'] = {'type': 'loss'}
                else:
                     log_msg = f"⚪ Skip (Score {last_pred['score']})"
                     stats['history'].append(log_msg)
                     delta['perf_change'] = {'type': 'skip'}

                # Update Streak
                if is_win:
                    st.session_state['streak_count'] += 1
                else:
                    st.session_state['streak_count'] = 0
        
        # Push to undo stack
        if 'undo_stack' in st.session_state:
            st.session_state['undo_stack'].append(delta)

    def undo_last_action():
        if st.session_state['current_game']:
            st.session_state['current_game'].pop()
            
        if 'undo_stack' in st.session_state and st.session_state['undo_stack']:
            delta = st.session_state['undo_stack'].pop()
            
            # Revert Mistakes
            if delta.get('added_mistake') is not None:
                # We assume it's the last one since it's a stack
                if st.session_state['session_mistakes']:
                    st.session_state['session_mistakes'].pop()
            
            # Revert Module Weights
            if delta.get('module_changes'):
                perf = st.session_state['module_performance']
                for name, change in delta['module_changes'].items():
                    perf[name] -= change # Reverse
            
            # Revert Streak
            if 'prev_streak' in delta:
                st.session_state['streak_count'] = delta['prev_streak']
            
            # Revert Win/Loss Stats
            if delta.get('perf_change'):
                p_change = delta['perf_change']
                stats = st.session_state['ai_performance']
                if p_change['type'] == 'win':
                    stats['wins'] -= 1
                elif p_change['type'] == 'loss':
                    stats['losses'] -= 1
                
                if stats['history']:
                    stats['history'].pop()

    if c1.button("🔵 PLAYER", use_container_width=True):
        update_learning(0)
        st.session_state['current_game'].append(0)
        st.rerun()
    if c2.button("🔴 BANKER", use_container_width=True):
        update_learning(1)
        st.session_state['current_game'].append(1)
        st.rerun()
    if c3.button("🟢 TIE", use_container_width=True):
        st.session_state['current_game'].append(2)
        # Ties don't update learning usually, but we push empty delta to keep stack sync?
        # Actually update_learning(2) handles it gracefully (no win/loss, no module change usually)
        # But wait, grade_module returns early on Tie.
        # Let's call update_learning(2) to keep stack synced.
        update_learning(2) 
        st.rerun()
    if c4.button("ลบ", type="primary"):
        undo_last_action()
        st.rerun()

with col2:
    st.subheader("🔮 3. Council Voting (Ensemble)")
    

    risk_thresholds = {"Low": 4, "Medium": 3, "High": 2}
    risk_level = st.session_state.get('risk_level', 'Medium')
    required_score = risk_thresholds[risk_level]
    

    mod_perf = st.session_state.get('module_performance', None)
    

    if mod_perf:
        st.caption("🧠 AI Thinking Process (Performance in this session):")
        cols = st.columns(4)
        emojis = {'historian': '📜', 'technician': '🛣️', 'statistician': '🧠', 'expert': '🎲'}
        for idx, (k, v) in enumerate(mod_perf.items()):
            emoji = emojis.get(k, '')
            color = "green" if v > 0 else ("red" if v < 0 else "gray")
            with cols[idx]:
                st.markdown(f"<div style='text-align:center; font-size:12px;'>{emoji}<br><span style='color:{color}; font-weight:bold'>{v:+d}</span></div>", unsafe_allow_html=True)
        st.divider()

    if 'models' in st.session_state and len(st.session_state['current_game']) >= 5:
        curr_hist = st.session_state['current_game']
        models = st.session_state['models']
        
        # Get ensemble prediction with dynamic weights
        prediction, score, vote_details = ensemble_predict(curr_hist, models, module_performance=mod_perf)
        
        if prediction is not None:
            result_text = "🔵 PLAYER" if prediction == 0 else "🔴 BANKER"
            
            # ========== Vote Breakdown ==========
            st.markdown("#### 📊 Vote Breakdown")
            
            # Create vote display
            vote_html = '<div style="background:#1a1a2e; padding:12px; border-radius:10px; margin-bottom:10px;">'
            
            # Historian (KNN)
            hist = vote_details.get('historian', {})
            hist_color = '#2196F3' if hist.get('vote') == 'PLAYER' else ('#f44336' if hist.get('vote') == 'BANKER' else '#888')
            vote_html += f'<div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #333;">'
            vote_html += f'<span style="color:#fff;">{hist.get("emoji", "📜")} Historian (KNN)</span>'
            vote_html += f'<span style="color:{hist_color}; font-weight:bold;">{hist.get("vote", "N/A")} ({hist.get("conf", 0):.0f}%)</span>'
            vote_html += '</div>'
            
            # Technician (Roads - Pattern Recognition)
            tech = vote_details.get('technician', {})
            tech_color = '#2196F3' if tech.get('vote') == 'PLAYER' else ('#f44336' if tech.get('vote') == 'BANKER' else '#888')
            vote_html += f'<div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #333;">'
            vote_html += f'<span style="color:#fff;">{tech.get("emoji", "🛣️")} Technician (Patterns)</span>'
            vote_html += f'<span style="color:{tech_color}; font-weight:bold;">{tech.get("vote", "N/A")} (Score: {tech.get("roads", 0)})</span>'
            vote_html += '</div>'
            
            # Statistician (RF)
            stat = vote_details.get('statistician', {})
            stat_color = '#2196F3' if stat.get('vote') == 'PLAYER' else ('#f44336' if stat.get('vote') == 'BANKER' else '#888')
            vote_html += f'<div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #333;">'
            vote_html += f'<span style="color:#fff;">{stat.get("emoji", "🧠")} Statistician (RF)</span>'
            vote_html += f'<span style="color:{stat_color}; font-weight:bold;">{stat.get("vote", "N/A")} ({stat.get("conf", 0):.0f}%)</span>'
            vote_html += '</div>'
            
            # Expert (Custom Patterns)
            expert = vote_details.get('expert', {})
            expert_color = '#2196F3' if expert.get('vote') == 'PLAYER' else ('#f44336' if expert.get('vote') == 'BANKER' else '#888')
            pattern_name = expert.get('pattern', '')
            pattern_display = f"✓ {pattern_name}" if pattern_name else expert.get('vote', 'N/A')
            vote_html += f'<div style="display:flex; justify-content:space-between; padding:5px 0;">'
            vote_html += f'<span style="color:#fff;">{expert.get("emoji", "🎲")} Expert (Your Patterns)</span>'
            vote_html += f'<span style="color:{expert_color}; font-weight:bold;">{pattern_display}</span>'
            vote_html += '</div>'
            

            
            vote_html += '</div>'
            st.markdown(vote_html, unsafe_allow_html=True)

            # --- Feature: Matched Pattern Alert ---
            if expert.get('pattern'):
                pat_name = expert['pattern'].replace('baccarat_results_', '').replace('.txt', '')
                st.warning(f"⚠️ **Pattern Alert:** ระบบตรวจพบเค้าไพ่ **\"{pat_name}\"** (Expert Module ยืนยัน)")
            
            # ========== Council Decision ==========
            st.markdown("#### 👥 Council Decision")
            
            # Score bar (max is now 6: KNN=2, Roads=1, RF=1, Expert=2)
            max_score = 6
            score_pct = min(score / max_score * 100, 100)
            
            # --- Feature: Confidence Meter ---
            meter_color = "#ccc" # Gray
            if score_pct >= 50: meter_color = "#2196F3" if prediction == 0 else "#f44336" # Blue/Red
            if score_pct >= 80: meter_color = "#FFD700" # Gold
            
            shadow = f"box-shadow: 0 0 10px {meter_color};" if score_pct >= 80 else ""
            
            st.markdown(f"""
            <div style="background:#333; height:12px; border-radius:6px; margin-bottom:5px;">
                <div style="width:{score_pct}%; background:{meter_color}; height:100%; border-radius:6px; transition: width 0.5s; {shadow}"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; color:#aaa; margin-bottom:15px;">
                <span>Confidence</span>
                <span>{score_pct:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Total Score: **{score}** / {max_score} (Need ≥ {required_score} to bet)")
            
            # Final recommendation based on risk level
            if score >= required_score:
                if score >= 4:
                    st.success(f"🎯 **STRONG BET: {result_text}**")
                    st.balloons()
                elif score >= 3:
                    st.success(f"✅ **RECOMMENDED: {result_text}**")
                else:
                    st.warning(f"⚠️ **MODERATE BET: {result_text}**")
            else:
                st.info(f"⏸️ **SKIP** - Score too low ({score} < {required_score})")
                st.caption(f"If betting anyway: {result_text}")
            
            # ========== Trap Warnings ==========
            st.divider()
            derived_features = get_derived_roads_features(curr_hist)
            current_streak = calculate_streak(curr_hist)
            tie_rate, _ = calculate_tie_stats(curr_hist, 20)
            
            warnings = []
            if derived_features['is_volatile']:
                warnings.append("⚠️ Roads เหวี่ยง - อาจเป็น Trap")
            if current_streak > 6:
                warnings.append(f"🐉 Dragon {current_streak} ตา - ระวังหัก")
            if tie_rate > 0.15:
                warnings.append(f"🟢 Tie สูง ({tie_rate*100:.0f}%)")
            
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.caption("✅ No trap signals detected")
            
            # Save prediction for next turn's learning
            st.session_state['last_prediction'] = {
                'vote': prediction,
                'score': score
            }
            st.session_state['last_vote_details'] = vote_details
                
        else:
            st.info("Need more non-tie hands (≥5)")
            
    elif 'models' not in st.session_state:
        st.info("👈 กรุณากดเทรน Ensemble ที่เมนูซ้าย")
    else:
        st.info("รอข้อมูลอีกนิด (ขออย่างน้อย 5 ตา)")


st.divider()
col_save1, col_save2 = st.columns([1, 1])

with col_save1:
    if st.button("💾 บันทึก & จบขอนนี้", use_container_width=True):
        if st.session_state['current_game']:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{timestamp}.txt"
            filepath = os.path.join("./data", filename)
            
            # Ensure data directory exists
            os.makedirs("./data", exist_ok=True)
            
            # Save the game data
            data_str = ", ".join(str(x) for x in st.session_state['current_game'])
            with open(filepath, 'w') as f:
                f.write(data_str)
            
            st.success(f"✅ บันทึกแล้ว: {filename}")
            st.session_state['current_game'] = []  # Reset game
            st.rerun()
        else:
            st.warning("ยังไม่มีข้อมูลให้บันทึก")

with col_save2:
    if st.button("🗑️ ล้างข้อมูลทั้งหมด", use_container_width=True, type="secondary"):
        st.session_state['current_game'] = []
        st.rerun()