import matplotlib,random
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False  
import numpy as np
def gen_trend(length, trend_type='stable', change_type='random', start=None, end=None):
    if trend_type == 'random': 
        trend_type = random.choices(['stable', 'up', 'down'], [0.2, 0.4, 0.4])[0]
    if start is None:
        if trend_type == 'up':
            start = random.uniform(0.1, 0.4)
        elif trend_type == 'down':
            start = random.uniform(0.6, 0.9)
        else:
            start = random.uniform(0.3, 0.7)

    if end is None:
        if trend_type == 'up':
            end = random.uniform(0.6, 0.9)
        elif trend_type == 'down':
            end = random.uniform(0.1, 0.4)
        else:
            end = start
        
    if trend_type == 'stable':
        if start > end:
            start, end = end, start
    elif trend_type == 'up':
        if start >= end:
            end = start + random.uniform(0.1, 0.5)
            if end > 1:
                end = 1
    elif trend_type == 'down':
        if start <= end:
            end = start - random.uniform(0.1, 0.5)
            if end < 0:
                end = 0
    else:
        raise ValueError("trend_type must be in ['stable', 'up', 'down']")
    if change_type == 'random':
        change_type = random.choice(['linear', 'quadratic', 'exponential'])
    if trend_type == 'stable':
        change_type = 'none'
    x = np.arange(length)
    meta = {
        'trend_type': trend_type,
        'change_type': change_type,
        'change_type_zh': {'linear': '线性', 'quadratic': '二次', 'exponential': '指数', 'none': 'none'}[change_type],
        'text_zh' : "整体趋势呈" + {'linear': '线性', 'quadratic': '二次', 'exponential': '指数', 'none':''}[change_type] + {'stable': '平稳' , 'up': '上升', 'down': '下降'}[trend_type] + "趋势",
        'text': "The overall trend in this interval is " + {'linear': 'linear', 'quadratic': 'quadratic', 'exponential': 'exponential', 'none':''}[change_type] + {'stable': 'stable' , 'up': 'upward', 'down': 'downward'}[trend_type] + " trend",
        'start': start,
        'end': end
    }
    if change_type == 'linear':
        trend = np.linspace(start, end, length)
    elif change_type == 'quadratic':
        a = (end - start) / (length ** 2)
        meta['change_alpha'] = a
        trend = a * (x ** 2) + start
    elif change_type == 'exponential':
        if start <= 0 or end <= 0:
            start += 0.1
            end += 0.1
        a = (end / start) ** (1 / length)
        meta['change_alpha'] = a
        trend = start * (a ** x)
    else :
        trend = np.full(length, start)
    noise = np.random.normal(0, 0.005, length)  # 添加噪声
    if trend_type == 'stable':
        noise = np.zeros(length)
    trend = trend + noise
    return trend, meta

def gen_season(length, season_type='stable', period=None, amplitude=None):
    """
    # 生成季节性成分
    - season_type in ['stable', 'square', 'sine']
    """
    if period is None:
        period_num = random.randint(3, 6)
        period = length // period_num
    if amplitude is None:
        amplitude = random.uniform(0.1, 0.5)
    x = np.arange(length)
    if season_type == 'random': 
        season_type = random.choice(['stable', 'square', 'sine'])
    # 保证起点终点为0
    if season_type == 'stable':
        season = np.zeros(length)
        text_zh = "无明显周期性变化"
        text = "No significant seasonal variation"
    elif season_type == 'square':
        # 保证season[0]是0
        season = amplitude * ((x % period) > (period / 2)).astype(float)
        text_zh = f"具有周期为{period}的方波周期特征"
        text = f"The seasonal variation is a square wave with a period of {period}"
    elif season_type == 'sine':
        season = amplitude * (np.sin(2 * np.pi * x / period)) / 2
        text_zh = f"具有周期为{period}的正弦波周期特征"
        text = f"The seasonal variation is a sine wave with a period of {period}"
    else:
        raise ValueError("season_type must be in ['stable', 'square', 'sine']")
    meta = {
        'season_type': season_type,
        'period': period,
        'amplitude': amplitude,
        'text_zh': text_zh,
        'text': text
    }
    return season, meta
def gen_segment(length, segment_type="none", start_point=0):
    seg = np.zeros(length)
    seg_scale = np.random.uniform(0.2, 0.5)
    seg_length = int(np.ceil(length * seg_scale))
    start_idx = random.randint(0, length - seg_length - 1)
    end_idx = start_idx + seg_length
    if segment_type == 'random':
        segment_type = random.choice(['up', 'down', 'volatile', 'platform'])
    if segment_type == 'none':
        meta = {
            'segment_type': 'none',
            'text_zh': "无明显的分段变化",
            'text': "No significant segment variation"
        }
        return seg, meta
    if segment_type == 'up':
        seg[start_idx : end_idx] = np.linspace(0, random.uniform(0.7, 1), seg_length)
        seg[start_idx] = 0
        seg[end_idx:] = seg[end_idx-1]
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point} 个点内，数值保持上升"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows an accelerating upward trend"
    elif segment_type == 'down':
        seg[start_idx : end_idx] = np.linspace(0, -random.uniform(0.7, 1), seg_length)
        seg[start_idx] = 0
        seg[end_idx :] = seg[end_idx-1]
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point}) 个点内，数值保持下降"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows an accelerating downward trend"
    elif segment_type == 'volatile':
        seg[start_idx : end_idx] = np.random.uniform(-1, 1, seg_length)
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point} 个点内，数值呈剧烈波动"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows violent fluctuations"
    elif segment_type == 'platform':
        level = random.uniform(-0.5, 0.5)
        seg[start_idx : end_idx] = level
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point} 个点内，数值呈平台状态，数值整体{'上升' if level>0 else '下降' if level<0 else ''}。"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows a platform state, level {'rising' if level>0 else 'falling' if level<0 else ''}, with a change of about {level:.2f}"
    else:
        raise ValueError("segment_type must be in ['none', 'up', 'down', 'volatile', 'platform']")
    meta = {
        'segment_type': segment_type,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'text_zh': text_zh,
        'text': text
    }
    return seg, meta
def gen_spike(length, spike_type="none", start_point=0, spike_num=1):
    spike = np.zeros(length)
    if spike_type == 'none':
        meta = {
            'spike_type': 'none',
            'text_zh': "无明显的突发变化",
            'text': "No significant spike variation"
        }
        return spike, meta
    spike_indices = random.sample(range(length), spike_num)
    spike_indices = sorted(spike_indices)
    text_zh = ''
    text = ''
    if spike_type == 'random':
        spike_type = [random.choice(['up', 'down']) for _ in range(spike_num)]
    elif spike_type in ['up', 'down']:
        spike_type = [spike_type] * spike_num
    for idx in spike_indices:
        #print(f'idx {idx} spike_type {spike_type}')
        spike_type_i = spike_type[spike_indices.index(idx)]
        if spike_type_i == 'up':
            spike[idx] = random.uniform(0.5, 1)
            text_zh += f"在第 {idx+start_point} 个点出现突发上升。"
            text += f"At point {idx+start_point}, the value shows a sudden upward spike."
        elif spike_type_i == 'down':
            spike[idx] = -random.uniform(0.5, 1)
            text_zh += f"在第 {idx+start_point} 个点出现突发下降。"
            text += f"At point {idx+start_point}, the value shows a sudden downward spike."
        else:
            raise ValueError("spike_type must be in ['none', 'up', 'down']")    
    meta = {
        'spike_type': spike_type,
        'spike_indices': [idx + start_point for idx in spike_indices],
        'text_zh': "",
        'text': ""
    }
    meta['text_zh'] = text_zh if text_zh else "无明显的突发变化"
    meta['text'] = text if text else "No significant spike variation"
    return spike, meta
def visual_func(length, func:list):
    import matplotlib.pyplot as plt
    x = np.arange(length)
    if len(func) > 10:
        print(f'func shape: {np.array(func).shape}')
    for i in func:
        y = []
        for t in range(length):
            y.append(i[t])
        plt.plot(y, label=f'{func.index(i)+1}')
    plt.legend() 
    plt.show()
def visual_timeseries(length, func: list, save_path: str = None, legend: bool = False, color: list = None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams   
    """
    Plot time series with hand-drawn style for ICLR 2026, no borders or axis labels, 4:3 aspect ratio.
    
    Args:
        length (int): Length of the time series.
        func (list): List of time series data.
        save_path (str, optional): Path to save the figure.
    """
    # Set figure size for 4:3 aspect ratio
    fig = plt.figure(figsize=(8, 6), dpi=300)
    
    # Remove spines and ticks
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Configure hand-drawn style
    rcParams['lines.linewidth'] = 2.5
    rcParams['lines.solid_capstyle'] = 'round'
    plt.style.use('seaborn-v0_8-deep')  # Use seaborn-deep for ICLR-friendly colors
    
    # Generate x-axis
    x = np.arange(length)
    
    # Check if func is too large and print shape
    if len(func) > 10:
        print(f'func shape: {np.array(func).shape}')
    
    # Plot each time series with hand-drawn style
    
    for idx, series in enumerate(func):
        y = np.array([series[t] for t in range(length)])
        if color and idx < len(color):
            idp = color[idx]
            plt.plot(x, y, label=f'TS {idx+1}', alpha=0.8, linewidth=4, color=plt.get_cmap('tab10')(idp % 10))
        else:
            plt.plot(x, y, label=f'TS {idx+1}', alpha=0.8, linewidth=4, color=plt.get_cmap('tab10')(idx % 10))
    # 背景透明
    plt.gca().set_facecolor('none')
    plt.tight_layout()
    # 让legend变大
    if legend:
        plt.legend(fontsize='large')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True, dpi=300)
    plt.show()