import streamlit as st
import pandas as pd

def render_big_road(history, mini=False):
    """เรนเดอร์ตาราง Big Road พร้อมรองรับ Mobile Scrolling"""
    ROWS = 6
    min_cols = 30 if not mini else 12
    COLS = max(min_cols, len(history) + 2)
    grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
    
    p_count = history.count(0)
    b_count = history.count(1)
    t_count = history.count(2)
    total = len(history)
    
    col, row, start_col = 0, 0, 0
    prev_winner = None
    is_tailing = False
    
    for item in history:
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
            start_col = 0
            is_tailing = False
        elif winner == prev_winner:
            if col >= COLS: break
            if not is_tailing:
                next_row = row + 1
                if next_row < ROWS and grid[next_row][col] is None:
                    row = next_row
                    grid[row][col] = {'color': color}
                else:
                    is_tailing = True
                    col += 1
                    if col < COLS: grid[row][col] = {'color': color}
            else:
                col += 1
                if col < COLS: grid[row][col] = {'color': color}
        else:
            start_col += 1
            col = start_col
            row = 0
            is_tailing = False
            while col < COLS and grid[row][col] is not None:
                col += 1
                start_col = col
            if col < COLS: grid[row][col] = {'color': color}
            prev_winner = winner

    cell_size = "28px" if not mini else "18px"
    circle_size = "20px" if not mini else "14px"
    font_size = "14px" if not mini else "11px"
    
    bg_color = "#ffffff" 
    border_color = "#e0e0e0"
    
    container_bg = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)" if not mini else "transparent"
    container_padding = "15px" if not mini else "0"
    container_radius = "12px" if not mini else "0"
    
    html = f"""
<style>
.baccarat-container {{ background: {container_bg}; padding: {container_padding}; border-radius: {container_radius}; margin-bottom: 15px; color: white; }}
.stats-bar {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 10px; border: 1px solid #333; flex-wrap: wrap; gap: 5px; }}
.stat-item {{ text-align: center; color: #ccc; font-size: {font_size}; min-width: 40px; }}
.stat-value {{ font-size: {20 if not mini else 12}px; font-weight: bold; color: #fff; }}
.stat-label {{ font-size: 10px; opacity: 0.6; }}
/* Mobile Scrolling */
.grid-wrapper {{ overflow-x: auto; background: {bg_color}; border-radius: 4px; padding: 2px; border: 1px solid {border_color}; width: 100%; box-sizing: border-box; -webkit-overflow-scrolling: touch; }}
.big-road-table {{ border-collapse: collapse; margin: 0 auto; min-width: 100%; }}
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
{'<div class="stat-item"><div class="stat-label" style="color:#448AFF">Player</div><div class="stat-value" style="color:#448AFF">' + str(p_count) + '</div></div>' if not mini else ''}
{'<div class="stat-item"><div class="stat-label" style="color:#FF5252">Banker</div><div class="stat-value" style="color:#FF5252">' + str(b_count) + '</div></div>' if not mini else ''}
{'<div class="stat-item"><div class="stat-label" style="color:#69F0AE">Tie</div><div class="stat-value" style="color:#69F0AE">' + str(t_count) + '</div></div>' if not mini else ''}
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

def render_raw_data(history):
    """เรนเดอร์ข้อมูลดิบแบบย่อสำหรับ Mobile"""
    if not history:
        return '<div style="background:#1a1a2e; color:#888; padding:20px; border-radius:12px; text-align:center; margin-bottom:15px;">ยังไม่มีข้อมูล</div>'
    
    mapping = {0: ('P', '#2196F3'), 1: ('B', '#f44336'), 2: ('T', '#4CAF50')}
    items_html = ''
    for i, val in enumerate(history):
        label, color = mapping.get(val, ('?', '#888'))
        items_html += f'<span style="display:inline-block; margin:2px; padding:4px 8px; background:{color}; color:#fff; border-radius:4px; font-weight:bold; font-size:11px;">{i+1}:{label}</span>'
    
    return f'''
    <div style="background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding:10px; border-radius:12px; margin-bottom:15px;">
        <div style="background:rgba(255,255,255,0.05); padding:8px; border-radius:8px; max-height:150px; overflow-y:auto; -webkit-overflow-scrolling: touch;">
            {items_html}
        </div>
        <div style="color:#888; font-size:10px; margin-top:5px; text-align:center;">รวม {len(history)} ตา | 0=P, 1=B, 2=T</div>
    </div>
    '''

def show_performance_dashboard(stats, streak, p_pct, b_pct):
    """แสดง Dashboard สรุปผลการเล่น"""
    wins = stats['wins']
    losses = stats['losses']
    total = wins + losses
    win_rate = (wins/total*100) if total > 0 else 0
    
    st.markdown(f"""
    <div style="background:#2d2d44; padding:15px; border-radius:12px; margin-bottom:15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid #444;">
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 10px; margin-bottom:15px; text-align:center;">
            <div><div style="color:#4CAF50; font-weight:bold; font-size:16px;">{wins}</div><div style="font-size:10px; opacity:0.6;">WIN</div></div>
            <div><div style="color:#f44336; font-weight:bold; font-size:16px;">{losses}</div><div style="font-size:10px; opacity:0.6;">LOSS</div></div>
            <div><div style="color:#FFC107; font-weight:bold; font-size:16px;">{streak}</div><div style="font-size:10px; opacity:0.6;">STREAK</div></div>
            <div><div style="color:#00BCD4; font-weight:bold; font-size:16px;">{win_rate:.0f}%</div><div style="font-size:10px; opacity:0.6;">RATE</div></div>
        </div>
        <div style="display:flex; height:8px; border-radius:4px; overflow:hidden; background:#333;">
            <div style="width:{p_pct}%; background:#2196F3;"></div>
            <div style="width:{b_pct}%; background:#f44336;"></div>
        </div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:5px; font-size:10px;">
            <span style="color:#2196F3">PLAYER {p_pct:.0f}%</span>
            <span style="color:#f44336">BANKER {b_pct:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Pattern Validator Modals ---
def get_modal_decorator():
    return getattr(st, "dialog", getattr(st, "experimental_dialog", None))

def show_gallery_modal(patterns):
    """แสดง Gallery ของ Pattern ทั้งหมด"""
    decorator = get_modal_decorator()
    if not decorator:
        st.error("Please update Streamlit to use Dialogs")
        return

    @decorator("🔍 Pattern Gallery", width="large")
    def _render(patterns):
        st.caption(f"📂 ทั้งหมด {len(patterns)} Patterns")
        cols = st.columns(min(len(patterns), 3) if patterns else 1)
        for idx, p in enumerate(patterns):
            with cols[idx % 3]:
                grid_html = render_big_road(p['pattern'], mini=True)
                predict = "???"
                if p.get('expected') == 0: predict = "<span style='color:#448AFF'>PLAYER</span>"
                elif p.get('expected') == 1: predict = "<span style='color:#FF5252'>BANKER</span>"
                
                display_name = p['name']
                st.markdown(f"""
                <div style="margin-bottom:16px; border:1px solid #444; border-radius:12px; padding:15px; background:#1e1e24; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <div style="font-size:12px; color:#aaa;">📄 {display_name}</div>
                        <div style="font-size:12px; font-weight:bold;">{predict}</div>
                    </div>
                    {grid_html}
                </div>
                """, unsafe_allow_html=True)
    _render(patterns)

def show_stats_modal(patterns, data_folder):
    """วิเคราะห์สถิติของแต่ละ Pattern จากข้อมูลย้อนหลัง"""
    decorator = get_modal_decorator()
    if not decorator:
        st.error("Please update Streamlit to use Dialogs")
        return

    @decorator("📊 Pattern Statistics", width="large")
    def _render(patterns, data_folder):
        from utils import load_all_games
        st.info("กำลังวิเคราะห์ Pattern จาก Data ทั้งหมด...")
        all_games = load_all_games(data_folder)
        if not all_games:
            st.error("ไม่พบข้อมูลเกมเพื่อใช้วิเคราะห์")
            return

        results = []
        progress_bar = st.progress(0)
        for idx, p in enumerate(patterns):
            pat_seq = p['pattern']
            pat_len = len(pat_seq)
            found, next_p, next_b = 0, 0, 0
            
            for game in all_games:
                for i in range(len(game) - pat_len):
                    if game[i:i+pat_len] == pat_seq:
                        found += 1
                        if i+pat_len < len(game):
                            nxt = game[i+pat_len]
                            if nxt == 0: next_p += 1
                            elif nxt == 1: next_b += 1
            
            total_valid = next_p + next_b
            p_rate = (next_p / total_valid * 100) if total_valid > 0 else 0
            b_rate = (next_b / total_valid * 100) if total_valid > 0 else 0
            
            rec = "⚪ 50/50"
            if p_rate > 55: rec = "🔵 PLAYER"
            elif b_rate > 55: rec = "🔴 BANKER"
            
            results.append({
                "ชื่อเค้าไพ่": p['name'],
                "เจอ (ครั้ง)": found,
                "ออก Player%": f"{p_rate:.1f}%",
                "ออก Banker%": f"{b_rate:.1f}%",
                "แนะนำ": rec
            })
            progress_bar.progress((idx + 1) / len(patterns))
            
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.success(f"วิเคราะห์เสร็จสิ้น {len(patterns)} รายการ")
    _render(patterns, data_folder)
