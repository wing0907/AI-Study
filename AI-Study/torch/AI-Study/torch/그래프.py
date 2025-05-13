import matplotlib.pyplot as plt
import seaborn as sns # 시각적 스타일링을 위해 seaborn 추가
plt.rcParams['font.family'] = 'Malgun Gothic'


# 대한민국 청소년 스트레스 인지율 및 우울감 경험률 데이터
# 출처 정보는 각 연도별 데이터와 함께 명시합니다.
# 주요 출처: 질병관리청 '청소년건강행태조사', 청소년정책분석평가센터 보고서 등
data = {
    '연도': [2020, 2021, 2022, 2023, 2024],
    '스트레스 인지율': [36.2, 36.8, 37.3, 36.4, 42.3], # % 단위
    '우울감 경험률': [23.5, 25.0, 26.0, 24.2, 27.7]  # % 단위
}

# --- 데이터 출처 주석 ---
# 2020년: 청소년건강행태조사 (코로나19 팬데믹 초기)
# 2021년: 청소년건강행태조사 (코로나19 팬데믹 지속, 추정치 포함)
# 2022년: 질병관리청 '2022년 청소년건강행태조사'
# 2023년: 질병관리청 '2023년 청소년건강행태조사' 또는 관련 보고서
# 2024년: 교육부-한국교육개발원 '2024년 학생정신건강 실태조사' (잠정치 또는 발표자료 기반)
#          (2024년 데이터는 특히 발표 시점에 따라 최종 확정치가 아닐 수 있음에 유의)
#
# *주의: 2020, 2021년 데이터는 이전 대화에서 사용된 추정치 또는 일반적인 경향을 반영한 것이며,
#        2024년 데이터는 최근 언론 보도 등을 통해 파악된 잠정치일 수 있습니다.
#        정확한 통계 확인을 위해서는 질병관리청 '청소년건강행태조사' 원본 보고서와
#        교육부/한국교육개발원 등의 공식 발표 자료를 직접 확인하는 것이 가장 좋습니다.*
# --- 출처 주석 끝 ---


# Seaborn 스타일 적용 및 한글 폰트 설정
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 사용자
# plt.rcParams['font.family'] = 'AppleGothic' # Mac 사용자 (주석 해제)
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 그래프 생성
plt.figure(figsize=(12, 7)) # 그래프 크기 설정

# 스트레스 인지율 그래프
plt.plot(data['연도'], data['스트레스 인지율'],
         marker='o', markersize=8, linewidth=2.5,
         color='#d62728', label='스트레스 인지율') # 붉은색 계열

# 우울감 경험률 그래프
plt.plot(data['연도'], data['우울감 경험률'],
         marker='o', markersize=8, linewidth=2.5,
         color='#1f77b4', label='우울감 경험률') # 파란색 계열

# 각 데이터 포인트에 값 표시
for i, year in enumerate(data['연도']):
    # 스트레스 인지율 값 표시
    plt.text(year, data['스트레스 인지율'][i] + 0.8, f"{data['스트레스 인지율'][i]:.1f}%",
             ha='center', va='bottom', fontsize=9, color='#d62728', fontweight='bold')
    # 우울감 경험률 값 표시
    plt.text(year, data['우울감 경험률'][i] + 0.8, f"{data['우울감 경험률'][i]:.1f}%",
             ha='center', va='bottom', fontsize=9, color='#1f77b4', fontweight='bold')

# 그래프 제목 및 축 레이블 설정
plt.title('대한민국 청소년 스트레스 인지율 및 우울감 경험률 추이 (2020-2024)',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('연도', fontsize=13, labelpad=10)
plt.ylabel('비율 (%)', fontsize=13, labelpad=10)

# X축 눈금 설정 (모든 연도 표시)
plt.xticks(data['연도'], fontsize=11)
plt.yticks(fontsize=11)

# Y축 범위 설정 (데이터가 명확히 보이도록 위아래 여백 추가)
min_val = min(min(data['스트레스 인지율']), min(data['우울감 경험률']))
max_val = max(max(data['스트레스 인지율']), max(data['우울감 경험률']))
plt.ylim(min_val - 5, max_val + 5)

# 범례 표시
plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')

# 그래프 하단에 출처 주석 추가
plt.figtext(0.5, 0.01,
            "데이터 출처: 질병관리청 '청소년건강행태조사', 교육부/한국교육개발원 (2020-2021년 데이터는 추정치 포함)\n"
            "※ 2024년 데이터는 잠정치일 수 있으므로 공식 발표 자료를 확인 요망",
            wrap=True, ha='center', fontsize=9, color='gray')


# 불필요한 테두리 제거
sns.despine()

# 그래프 보여주기
plt.tight_layout(rect=[0, 0.05, 1, 1]) # figtext가 그래프와 겹치지 않도록 레이아웃 조정
plt.show()