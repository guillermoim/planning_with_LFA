import matplotlib.pyplot as plt


def __init__():
    pass

def max_score_by_parameter(df, features):
    
    fig, ax = plt.subplots(1,1, figsize=(9, 6))
    ax.set_title(f'# feat. = {features}')
    ax.plot(df.parameter.to_numpy(), df.score.to_numpy(), marker='d')
    ax.set_ylim((-0.5, 1.1))
    ax.set_ylabel('score - $R^2$')
    ax.set_xlabel('$\lambda$')
    ax.tick_params(axis='x', rotation=90)
    ax.set_xticks(df.parameter.to_numpy())
    ax.set_xscale("log")
    
def max_score_by_num_features(df, step=5):
    
    df = df.loc[df.groupby('total_features')["score"].idxmax()]
    fig, ax = plt.subplots(1,1, figsize=(9, 6))

    ax2 = ax.twinx()

    ax.set_title(f'Max. score (and opt. param.) by # of features')
    ax.plot(df.total_features.to_numpy(), df.score.to_numpy())
    
    ax.set_ylim((0.9, 1.001))
    ax.set_ylabel('score - $R^2$')
    ax.set_xlabel('# features considered')
    
    ax.set_xticks(df.total_features.to_numpy()[::step], rotation=-75)

    
    ax2.plot(df.total_features.to_numpy(), df.parameter.to_numpy(), color='red')
    ax2.set_yscale("log")
    ax2.set_ylabel('$\lambda$')
    ax2.tick_params(axis='x', labelbottom=False)
    
    
def support_by_lambda(df, num_features):
    
    df_ = df[df.total_features==num_features][["parameter", "max_support_size"]]
    fig, ax = plt.subplots(1,1, figsize=(9, 6))
    
    ax.set_title(f'Support size by $\lambda$ # feat. = {num_features}')
    ax.plot(df_.parameter.to_numpy(), df_.max_support_size.to_numpy(), marker='d')
    ax.set_ylabel('support size')
    ax.set_xlabel('$\lambda$')
    ax.tick_params(axis='x', rotation=90)
    ax.set_xticks(df_.parameter.to_numpy())
    ax.set_xscale("log")

    
    
    
    
    
    
    
    
    
    
    