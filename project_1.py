"""
🏭 총괄생산계획(APP) 최적화 웹앱
원예장비 제조업체의 총괄생산계획을 LP/IP로 최적화하고 시각화
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="총괄생산계획 최적화",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# 솔버
# ──────────────────────────────────────────────────────────
def solve_app(demand, params, method='IP'):
    T = len(demand)
    n = 8 * T

    def ix(var, t):
        return 8 * t + {'W':0,'H':1,'L':2,'P':3,'I':4,'S':5,'C':6,'O':7}[var]

    reg_cap = params['prod_rate'] * params['hrs_day'] * params['work_days']

    c = np.zeros(n)
    for t in range(T):
        c[ix('W',t)] = params['wage'];      c[ix('H',t)] = params['hire']
        c[ix('L',t)] = params['fire'];       c[ix('P',t)] = params['material']
        c[ix('I',t)] = params['inv_hold'];   c[ix('S',t)] = params['shortage']
        c[ix('C',t)] = params['outsource'];  c[ix('O',t)] = params['ot_wage']

    rows, lb, ub = [], [], []
    for t in range(T):
        r = np.zeros(n)
        r[ix('W',t)] = 1; r[ix('H',t)] = -1; r[ix('L',t)] = 1
        rhs = params['init_workers'] if t == 0 else 0
        if t > 0: r[ix('W',t-1)] = -1
        rows.append(r); lb.append(rhs); ub.append(rhs)

        r = np.zeros(n)
        r[ix('I',t)] = 1; r[ix('P',t)] = -1
        r[ix('C',t)] = -1; r[ix('S',t)] = -1
        if t > 0:
            r[ix('I',t-1)] = -1; r[ix('S',t-1)] = 1
            rhs = -demand[t]
        else:
            rhs = params['init_inv'] - demand[0]
        rows.append(r); lb.append(rhs); ub.append(rhs)

        r = np.zeros(n)
        r[ix('P',t)] = 1; r[ix('W',t)] = -reg_cap; r[ix('O',t)] = -params['prod_rate']
        rows.append(r); lb.append(-np.inf); ub.append(0)

        r = np.zeros(n)
        r[ix('O',t)] = 1; r[ix('W',t)] = -params['ot_limit']
        rows.append(r); lb.append(-np.inf); ub.append(0)

    r = np.zeros(n); r[ix('S',T-1)] = 1
    rows.append(r); lb.append(0); ub.append(0)
    r = np.zeros(n); r[ix('I',T-1)] = 1
    rows.append(r); lb.append(params['final_inv']); ub.append(np.inf)

    A = np.array(rows)

    if method == 'LP':
        eq_mask = [l == u for l, u in zip(lb, ub)]
        A_eq = A[eq_mask]; b_eq = np.array(lb)[eq_mask]
        A_ub_list, b_ub_list = [], []
        lb_arr = np.array(lb); ub_arr = np.array(ub)
        for i in range(len(lb)):
            if eq_mask[i]: continue
            if ub_arr[i] < np.inf:
                A_ub_list.append(A[i]); b_ub_list.append(ub_arr[i])
            if lb_arr[i] > -np.inf:
                A_ub_list.append(-A[i]); b_ub_list.append(-lb_arr[i])
        res = linprog(c, A_ub=np.array(A_ub_list), b_ub=np.array(b_ub_list),
                      A_eq=A_eq, b_eq=b_eq, bounds=[(0,None)]*n, method='highs')
    else:
        integ = np.zeros(n)
        for t in range(T):
            for v in ['W','H','L','P','I','S','C']:
                integ[ix(v,t)] = 1
        res = milp(c, constraints=LinearConstraint(A, lb, ub),
                   integrality=integ, bounds=Bounds(lb=0, ub=np.inf))

    if not res.success:
        return None

    x = res.x
    records = []
    for t in range(T):
        W=x[ix('W',t)]; H=x[ix('H',t)]; L=x[ix('L',t)]
        P=x[ix('P',t)]; I=x[ix('I',t)]; S=x[ix('S',t)]
        C_=x[ix('C',t)]; O=x[ix('O',t)]
        rc = reg_cap * W
        ot_prod = O * params['prod_rate']
        mc = (params['wage']*W + params['ot_wage']*O + params['hire']*H +
              params['fire']*L + params['inv_hold']*I + params['shortage']*S +
              params['material']*P + params['outsource']*C_)
        records.append({
            '월': t+1, '수요': int(demand[t]),
            '근로자': round(W,1), '고용': round(H,1), '해고': round(L,1),
            '생산량': round(P,1), '초과생산': round(ot_prod,1),
            '초과근무(hr)': round(O,1), '외주': round(C_,1),
            '재고': round(I,1), '부족': round(S,1),
            '총공급': round(P+C_,1), '정규능력': round(rc,1),
            '인건비': round(params['wage']*W,1),
            '초과근무비': round(params['ot_wage']*O,1),
            '고용비': round(params['hire']*H,1),
            '해고비': round(params['fire']*L,1),
            '재고유지비': round(params['inv_hold']*I,1),
            '부족비용': round(params['shortage']*S,1),
            '재료비': round(params['material']*P,1),
            '외주비': round(params['outsource']*C_,1),
            '월비용': round(mc, 1),
        })

    df = pd.DataFrame(records)
    total_cost = res.fun
    total_rev = params['sell_price'] * sum(demand)
    return {'df': df, 'total_cost': total_cost, 'total_revenue': total_rev,
            'profit': total_rev - total_cost, 'method': method}


# ──────────────────────────────────────────────────────────
# 색상
# ──────────────────────────────────────────────────────────
C = {
    'blue':'#2563eb', 'purple':'#7c3aed', 'green':'#059669',
    'orange':'#d97706', 'red':'#dc2626', 'cyan':'#0891b2',
    'pink':'#db2777', 'gray':'#475569',
}


# ──────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────
st.sidebar.title("🏭 파라미터 설정")

# 수요 입력 (동적 월 수)
st.sidebar.header("📊 수요 예측")
n_months = st.sidebar.slider("계획 기간 (개월)", 2, 12, 6)

default_demands = [1600, 3000, 3200, 3800, 2200, 2200, 2000, 2000, 2000, 2000, 2000, 2000]

# 프리셋
# 프리셋
preset = st.sidebar.selectbox("프리셋", ["6개월(default)", "직접 입력"])
if preset == "6개월(default)":
    default_demands = [1600, 3000, 3200, 3800, 2200, 2200] + [2000]*6

demand = []
cols = st.sidebar.columns(3)
for i in range(n_months):
    with cols[i % 3]:
        d = st.number_input(f"{i+1}월", 0, 50000,
                            default_demands[i] if i < len(default_demands) else 2000,
                            step=100, key=f"d_{i}")
        demand.append(d)

# 비용 파라미터
st.sidebar.header("💰 비용 파라미터")
c1, c2 = st.sidebar.columns(2)
with c1:
    wage = st.number_input("정규임금 (천원/인·월)", 0, 5000, 640)
    ot_wage = st.number_input("초과근무임금 (천원/hr)", 0, 100, 6)
    hire = st.number_input("고용비 (천원/인)", 0, 5000, 300)
    fire = st.number_input("해고비 (천원/인)", 0, 5000, 500)
with c2:
    inv_hold = st.number_input("재고유지비 (천원/개·월)", 0, 100, 2)
    shortage = st.number_input("부족비용 (천원/개·월)", 0, 100, 5)
    material = st.number_input("재료비 (천원/개)", 0, 500, 10)
    outsource = st.number_input("외주비 (천원/개)", 0, 500, 30)

sell_price = st.sidebar.number_input("판매단가 (천원/개)", 0, 1000, 40)

# 생산능력
st.sidebar.header("⚙️ 생산능력")
c3, c4 = st.sidebar.columns(2)
with c3:
    work_days = st.number_input("작업일수 (일/월)", 1, 31, 20)
    hrs_day = st.number_input("일 근무시간 (hr)", 1, 24, 8)
with c4:
    ot_limit = st.number_input("초과시간 제한 (hr/인·월)", 0, 100, 10)
    prod_rate = st.number_input("생산율 (개/hr)", 0.01, 10.0, 0.25, step=0.05, format="%.2f")

# 초기/최종 조건
st.sidebar.header("📦 초기·최종 조건")
c5, c6 = st.sidebar.columns(2)
with c5:
    init_workers = st.number_input("초기 근로자 (명)", 0, 500, 80)
    init_inv = st.number_input("초기 재고 (개)", 0, 50000, 1000)
with c6:
    final_inv = st.number_input("최종 재고 ≥ (개)", 0, 50000, 500)

# 풀이 방법
solve_method = st.sidebar.radio("풀이 방법", ["IP (정수계획법)", "LP (선형계획법)"], index=0)

params = {
    'wage': wage, 'ot_wage': ot_wage, 'hire': hire, 'fire': fire,
    'inv_hold': inv_hold, 'shortage': shortage, 'material': material,
    'outsource': outsource, 'sell_price': sell_price,
    'work_days': work_days, 'hrs_day': hrs_day, 'ot_limit': ot_limit,
    'prod_rate': prod_rate,
    'init_workers': init_workers, 'init_inv': init_inv, 'final_inv': final_inv,
}


# ──────────────────────────────────────────────────────────
# 최적화 실행
# ──────────────────────────────────────────────────────────
method = 'IP' if 'IP' in solve_method else 'LP'
result = solve_app(demand, params, method)

# ──────────────────────────────────────────────────────────
# 메인 영역
# ──────────────────────────────────────────────────────────
st.title("🏭 총괄생산계획 최적화")
st.caption("원예장비 제조업체 · Aggregate Production Planning Optimizer")

if result is None:
    st.error("⚠️ 최적화에 실패했습니다. 파라미터를 확인해 주세요.")
    st.stop()

df = result['df']
labels = [f"{i+1}월" for i in range(n_months)]
labels_ext = ['초기'] + labels

# KPI
st.success(f"✅ {result['method']} 최적해 도달 ({n_months}개월) · 총비용 **{result['total_cost']:,.0f}** 천원")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("총 비용", f"{result['total_cost']:,.0f}", "천원")
k2.metric("총 수익", f"{result['total_revenue']:,.0f}", "천원")
k3.metric("순이익", f"{result['profit']:,.0f}",
          f"{result['profit']/result['total_revenue']*100:.1f}%")
k4.metric("총 생산량", f"{df['생산량'].sum():,.0f}", "개")
k5.metric("평균 근로자", f"{df['근로자'].mean():.1f}", "명")
k6.metric("총 외주량", f"{df['외주'].sum():,.0f}", "개")

st.divider()

# 탭
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 종합 대시보드", "🔧 생산 분석", "👷 인력 관리",
    "📦 재고 분석", "💵 비용 분석"
])

# ────────── 탭 1: 종합 ──────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=labels, y=df['생산량'], name='자체생산', marker_color=C['blue'])
        fig.add_bar(x=labels, y=df['외주'], name='외주생산', marker_color=C['orange'])
        fig.add_scatter(x=labels, y=df['수요'], name='수요',
                        mode='lines+markers', line=dict(color=C['red'], width=3))
        fig.update_layout(title='생산량 vs 수요', barmode='stack',
                          height=350, template='plotly_white',
                          legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cost_items = {}
        for col in ['인건비','초과근무비','고용비','해고비','재고유지비','부족비용','재료비','외주비']:
            v = df[col].sum()
            if v > 0.5: cost_items[col] = v
        colors = [C['blue'],C['orange'],C['green'],C['red'],C['cyan'],C['pink'],C['gray'],C['purple']]
        fig = go.Figure(go.Pie(labels=list(cost_items.keys()), values=list(cost_items.values()),
                               hole=0.45, marker_colors=colors[:len(cost_items)]))
        fig.update_layout(title='비용 구성', height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_scatter(x=labels_ext, y=[init_workers]+list(df['근로자']),
                        fill='tozeroy', name='근로자',
                        fillcolor='rgba(124,58,237,0.15)',
                        line=dict(color=C['purple'], width=2.5))
        fig.add_bar(x=labels, y=df['고용'], name='고용', marker_color=C['green'])
        fig.add_bar(x=labels, y=df['해고'], name='해고', marker_color=C['red'])
        fig.update_layout(title='인력 변동', height=300, template='plotly_white',
                          legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = go.Figure()
        fig.add_scatter(x=labels_ext, y=[init_inv]+list(df['재고']),
                        fill='tozeroy', name='재고',
                        fillcolor='rgba(8,145,178,0.15)',
                        line=dict(color=C['cyan'], width=2.5))
        fig.add_bar(x=labels, y=df['부족'], name='부족', marker_color=C['red'])
        fig.add_hline(y=final_inv, line_dash='dash', line_color=C['orange'],
                      annotation_text=f"최종재고≥{final_inv}")
        fig.update_layout(title='재고 추이', height=300, template='plotly_white',
                          legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # 상세 테이블
    st.subheader("📋 월별 상세 계획")
    show_cols = ['월','수요','근로자','고용','해고','생산량','초과근무(hr)','외주','재고','부족','월비용']
    st.dataframe(
        df[show_cols].style.format({
            '근로자':'{:.1f}','고용':'{:.1f}','해고':'{:.1f}',
            '생산량':'{:.1f}','초과근무(hr)':'{:.1f}','외주':'{:.1f}',
            '재고':'{:.1f}','부족':'{:.1f}','월비용':'{:,.0f}',
        }).background_gradient(subset=['부족'], cmap='Reds', vmin=0)
         .background_gradient(subset=['월비용'], cmap='Blues', vmin=0),
        use_container_width=True, hide_index=True,
    )

# ────────── 탭 2: 생산 분석 ──────────
with tab2:
    fig = go.Figure()
    fig.add_bar(x=labels, y=df['생산량']-df['초과생산'], name='정규생산', marker_color=C['blue'])
    fig.add_bar(x=labels, y=df['초과생산'], name='초과생산', marker_color=C['orange'])
    fig.add_bar(x=labels, y=df['외주'], name='외주', marker_color=C['pink'])
    fig.add_scatter(x=labels, y=df['수요'], name='수요',
                    mode='lines+markers', line=dict(color=C['red'], width=3))
    fig.update_layout(title='생산 유형별 분석', barmode='stack',
                      height=400, template='plotly_white',
                      legend=dict(orientation='h', y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        util = []
        for _, r in df.iterrows():
            cap = r['정규능력']
            util.append(min(r['생산량'], cap) / cap * 100 if cap > 0 else 0)
        fig = go.Figure(go.Bar(x=labels, y=util,
            marker_color=[C['green'] if u>80 else C['orange'] if u>50 else C['red'] for u in util],
            text=[f"{u:.0f}%" for u in util], textposition='outside'))
        fig.add_hline(y=100, line_dash='dash', line_color='gray')
        fig.update_layout(title='정규 생산능력 활용률 (%)', height=350,
                          template='plotly_white', yaxis_range=[0, max(util)*1.2 if util else 100])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cum_s = np.cumsum([init_inv] + list(df['생산량']+df['외주']))
        cum_d = np.cumsum([0] + list(df['수요']))
        fig = go.Figure()
        fig.add_scatter(x=labels_ext, y=cum_s, name='누적 공급',
                        mode='lines+markers', line=dict(color=C['blue'], width=2.5))
        fig.add_scatter(x=labels_ext, y=cum_d, name='누적 수요',
                        mode='lines+markers', line=dict(color=C['red'], width=2.5))
        fig.update_layout(title='누적 생산 vs 누적 수요', height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ────────── 탭 3: 인력 관리 ──────────
with tab3:
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('근로자 수 추이','고용/해고','초과근무 시간','인력 관련 비용'))
    fig.add_scatter(x=labels_ext, y=[init_workers]+list(df['근로자']),
                    fill='tozeroy', name='근로자',
                    fillcolor='rgba(124,58,237,0.15)',
                    line=dict(color=C['purple'], width=2.5), row=1, col=1)
    fig.add_bar(x=labels, y=df['고용'], name='고용', marker_color=C['green'], row=1, col=2)
    fig.add_bar(x=labels, y=df['해고'], name='해고', marker_color=C['red'], row=1, col=2)
    fig.add_bar(x=labels, y=df['초과근무(hr)'], name='초과근무',
                marker_color=C['orange'], row=2, col=1)
    fig.add_bar(x=labels, y=df['인건비'], name='인건비', marker_color=C['blue'], row=2, col=2)
    fig.add_bar(x=labels, y=df['고용비'], name='고용비', marker_color=C['green'], row=2, col=2)
    fig.add_bar(x=labels, y=df['해고비'], name='해고비', marker_color=C['red'], row=2, col=2)
    fig.update_layout(height=650, template='plotly_white',
                      legend=dict(orientation='h', y=-0.05))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("총 고용", f"{df['고용'].sum():.0f}명", f"비용 {df['고용비'].sum():,.0f}천원")
    m2.metric("총 해고", f"{df['해고'].sum():.0f}명", f"비용 {df['해고비'].sum():,.0f}천원")
    m3.metric("총 초과근무", f"{df['초과근무(hr)'].sum():.0f}hr",
              f"비용 {df['초과근무비'].sum():,.0f}천원")

# ────────── 탭 4: 재고 분석 ──────────
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=labels_ext, y=[init_inv]+list(df['재고']),
                        fill='tozeroy', name='재고',
                        fillcolor='rgba(8,145,178,0.15)',
                        line=dict(color=C['cyan'], width=2.5))
        fig.add_hline(y=final_inv, line_dash='dash', line_color=C['orange'],
                      annotation_text=f"최종재고≥{final_inv}")
        fig.update_layout(title='재고 수준', height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        balance = df['총공급'] - df['수요']
        fig = go.Figure(go.Bar(x=labels, y=balance,
            marker_color=[C['green'] if b>=0 else C['red'] for b in balance],
            text=[f"{b:+.0f}" for b in balance], textposition='outside'))
        fig.add_hline(y=0, line_color='gray')
        fig.update_layout(title='월별 공급-수요 잔차', height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    if df['부족'].sum() > 0:
        st.warning(f"⚠️ 총 부족량: {df['부족'].sum():.0f}개 · 부족비용: {df['부족비용'].sum():,.0f}천원")
    else:
        st.success("✅ 모든 월의 수요를 완전히 충족합니다.")

# ────────── 탭 5: 비용 분석 ──────────
with tab5:
    cost_cols = ['인건비','초과근무비','고용비','해고비','재고유지비','부족비용','재료비','외주비']
    cost_colors = [C['blue'],C['orange'],C['green'],C['red'],C['cyan'],C['pink'],C['gray'],C['purple']]

    fig = go.Figure()
    for col, clr in zip(cost_cols, cost_colors):
        if df[col].sum() > 0:
            fig.add_bar(x=labels, y=df[col], name=col, marker_color=clr)
    fig.update_layout(title='월별 비용 구성', barmode='stack', height=400,
                      template='plotly_white', legend=dict(orientation='h', y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("비용 항목별 합계")
    summary = []
    for col in cost_cols:
        v = df[col].sum()
        if v > 0:
            summary.append({'항목':col, '금액(천원)':f"{v:,.0f}",
                            '비율':f"{v/result['total_cost']*100:.1f}%"})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.metric("총 비용 합계", f"{result['total_cost']:,.0f} 천원")

# ──────────────────────────────────────────────────────────
# 수학적 모델 설명
# ──────────────────────────────────────────────────────────
with st.expander("📐 수학적 모델 설명"):
    st.markdown(r"""
### 결정변수
| 변수 | 의미 | 단위 |
|:---|:---|:---|
| $W_t$ | $t$월의 종업원 수 | 인/월 |
| $H_t$ | $t$월초에 고용하는 종업원 수 | 인/월 |
| $L_t$ | $t$월초에 해고하는 종업원 수 | 인/월 |
| $P_t$ | $t$월의 생산량 | 개/월 |
| $I_t$ | $t$월 말의 재고 | 개/월 |
| $S_t$ | $t$월 말의 부족재고 | 개/월 |
| $C_t$ | $t$월에 하청 계약되는 제품의 수 | 개/월 |
| $O_t$ | $t$월에 작업된 총 초과시간 | 시간/월 |

### 목적함수 (비용 최소화)
$$\min Z = \sum_{t=1}^{T} \left( 640 W_t + 6 O_t + 300 H_t + 500 L_t + 2 I_t + 5 S_t + 10 P_t + 30 C_t \right)$$

### 제약조건
1. **노동력 균형**: $W_t = W_{t-1} + H_t - L_t$
2. **생산능력**: $P_t \leq 40 W_t + \frac{O_t}{4}$
3. **재고 균형**: $I_t = I_{t-1} + P_t + C_t - D_t - S_{t-1} + S_t$
4. **초과근무 제한**: $O_t \leq 10 W_t$
5. **비음수**: 모든 변수 $\geq 0$
6. **초기조건**: $W_0 = 80, I_0 = 1000, S_0 = 0$
7. **최종조건**: $I_T \geq 500, S_T = 0$
    """)
