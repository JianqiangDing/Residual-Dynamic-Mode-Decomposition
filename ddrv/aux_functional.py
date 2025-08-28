# auxiliary functions for ddrv

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint

# defination of dynamical systems ------------------------------------------------

# define the NL_EIG system (2D nonlinear system with known eigenvalues and eigenfunctions)


class DynamicalSystem(ABC):
    """
    abstract base class for defining dynamical systems
    """

    def __init__(self, dimension: int):
        """
        initialize the dynamical system

        Args:
            dimension: the dimension of the system
        """
        self.dimension = dimension

    @abstractmethod
    def get_dynamics(self) -> sp.Matrix:
        """
        get the dynamical system definition

        Returns:
            sp.Matrix: the dynamical system equation
        """
        pass

    def get_numerical_dynamics(self) -> Callable:
        """
        get the numerical dynamics function, for numerical integration

        Returns:
            Callable: the numerical dynamics function
        """
        # convert the symbolic expression to a numerical function
        f_symbolic = self.get_dynamics()
        x_vars = sp.symbols(f"x:{self.dimension}")
        f_lambdified = sp.lambdify(x_vars, f_symbolic, "numpy")

        def dynamics(t, x):
            return np.array(f_lambdified(*x)).flatten()

        return dynamics


@dataclass
class NL_EIG_System(DynamicalSystem):
    """
    NL_EIG system class - a 2D nonlinear system with known eigenvalues and eigenfunctions
    """

    lambda1: float = -1.0
    lambda2: float = 2.5

    def __post_init__(self):
        """initialize the system"""
        super().__init__(dimension=2)
        self.Lambda = np.array([self.lambda1, self.lambda2])

        # define the symbolic variables
        self.x = sp.symbols("x:2")

        # define the principal eigenfunctions
        self.psi1 = self.x[0] ** 2 + 2 * self.x[1] + self.x[1] ** 3
        self.psi2 = self.x[0] + sp.sin(self.x[1]) + self.x[0] ** 3

        self.Psi = sp.Matrix([self.psi1, self.psi2])

        # compute the Jacobian matrix
        self.J = self.Psi.jacobian(self.x)

        # define the dynamical system
        self.f = self.J.inv() @ np.diag(self.Lambda) @ self.Psi

    def get_dynamics(self):
        """
        get the dynamical system definition

        Returns:
            sp.Matrix: the dynamical system equation
        """
        return self.f

    def get_eigenfunctions(self):
        """
        get the eigenfunctions

        Returns:
            sp.Matrix: the eigenfunctions vector
        """
        return self.Psi

    def get_eigenvalues(self):
        """
        get the eigenvalues
        """
        return self.Lambda


# --------------------------------------------------------------------------------
# now some auxiliary functions for obtaining the trajectory data from the given dynamical system
def generate_trajectory_data(
    dynamical_system: DynamicalSystem,
    num_samples: int = 1000,
    num_steps: int = 10,
    delta_t: float = 0.05,
    domain: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate the trajectory data

    Args:
        dynamical_system: the dynamical system object
        num_samples: the number of trajectories
        num_steps: the number of time steps per trajectory
        delta_t: the time step
        domain: the domain of the initial conditions (min_val, max_val), if None then use (-1, 1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, Y) data pairs, X is the current state, Y is the next state
    """
    if domain is None:
        # domain should be an array of shape (dimension, 2), each row is the min and max of the domain for each dimension
        domain = np.array([(-1, 1)] * dynamical_system.dimension)
    else:
        domain = np.array(domain)
    assert domain.shape == (
        dynamical_system.dimension,
        2,
    ), "the domain must be a tuple of length the dimension of the system"

    # Vectorized integration for all initial points using scipy.integrate.odeint
    dim = dynamical_system.dimension

    # Draw all initial conditions at once, shape: (dim, M1)
    Y0_batch = (np.random.rand(dim, num_samples)) * (
        (domain[:, 1] - domain[:, 0]).reshape(dim, 1)
    ) + domain[:, 0].reshape(dim, 1)

    # Vectorized dynamics from symbolic definition
    f_symbolic = dynamical_system.get_dynamics()
    x_vars = sp.symbols(f"x:{dim}")
    f_vec = sp.lambdify(x_vars, f_symbolic, "numpy")

    # Batch ODE for odeint (flattened state of all points)
    def ode_fx(x_flat, t):
        X_pts = x_flat.reshape(num_samples, dim)  # (Npoints, dim)
        vals = f_vec(*(X_pts.T))  # returns (dim, Npoints) or similar
        V = np.array(vals)
        if V.ndim == 1:
            V = V.reshape(dim, -1)
        return V.T.flatten()

    # Small burn-in to avoid t=0
    T_min = 1e-6
    steps_init = int(np.ceil(T_min / delta_t)) if T_min > 0 else 0
    if steps_init > 0:
        x_init = odeint(
            ode_fx, Y0_batch.T.flatten(), np.linspace(0.0, T_min, steps_init)
        )[-1]
        X_cur = x_init.reshape(num_samples, dim)
    else:
        X_cur = Y0_batch.T  # (Npoints, dim)

    # March forward for num_steps steps with small internal solver grid
    traj = [X_cur]  # list of (Npoints, dim)
    t_small = np.linspace(0.0, delta_t, 5)
    for _ in range(num_steps):
        x_next = odeint(ode_fx, X_cur.flatten(), t_small)[-1].reshape(num_samples, dim)
        traj.append(x_next)
        X_cur = x_next

    print(
        f"generated {num_samples} trajectories in batch (odeint), {num_steps} steps, {delta_t} time step..."
    )

    return np.stack(traj)


def visualize_vector_field(dynamics, domain=[-2, 2, -2, 2], step_size=0.1):
    """
    visualize the vector field of the dynamical system
    """
    xmin, xmax, ymin, ymax = domain

    # create grid
    x = np.arange(xmin, xmax + step_size, step_size)
    y = np.arange(ymin, ymax + step_size, step_size)
    X, Y = np.meshgrid(x, y)

    # compute velocity field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                dxdt = dynamics(0, [X[i, j], Y[i, j]])
                U[i, j] = dxdt[0]
                V[i, j] = dxdt[1]
            except Exception:
                U[i, j] = np.nan
                V[i, j] = np.nan

    # plot streamlines
    plt.figure(figsize=(12, 9))

    start_points = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    start_points = np.array([start_points[0].flatten(), start_points[1].flatten()]).T

    plt.streamplot(
        X, Y, U, V, start_points=start_points, color="blue", linewidth=0.8, density=1.5
    )

    # add contour lines of speed for context
    speed = np.sqrt(U**2 + V**2)
    plt.contour(
        X, Y, speed, levels=8, colors="gray", linestyles="--", linewidths=0.5, alpha=0.7
    )

    # formatting
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Vector Field (Streamlines)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.show()


def visualize_scalar_function_3d_plotly(
    func_true=None,
    func_edmd=None,
    domain_x=(-2, 2),
    domain_y=(-2, 2),
    num_pts=100,
    title="标量函数3D可视化",
    true_eigenval=None,
    edmd_eigenval=None,
    show_fig=True,
    save_html=None,
    estimate_scaling=True,
):
    """
    使用plotly在2D域上可视化标量函数的3D表面图

    Args:
        func_true: 真实函数，接受(x, y)并返回标量值
        func_edmd: EDMD近似函数，接受(x, y)并返回标量值
        domain_x: x轴范围 (min, max)
        domain_y: y轴范围 (min, max)
        num_pts: 网格点数
        title: 图形标题
        true_eigenval: 真实特征值（用于标题显示）
        edmd_eigenval: EDMD特征值（用于标题显示）
        show_fig: 是否显示图形
        save_html: 如果提供文件路径，将图形保存为HTML文件
        estimate_scaling: 是否估计缩放因子来匹配真实函数

    Returns:
        plotly.graph_objects.Figure: plotly图形对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly库未安装。请安装: pip install plotly")

    # 创建网格
    xx = np.linspace(domain_x[0], domain_x[1], num_pts)
    yy = np.linspace(domain_y[0], domain_y[1], num_pts)
    xx, yy = np.meshgrid(xx, yy)

    # 初始化数据列表
    data = []

    # 计算EDMD函数值（如果提供）
    if func_edmd is not None:
        zz_edmd = func_edmd(xx, yy)

        # 添加EDMD散点数据
        data.append(
            go.Scatter3d(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz_edmd.flatten(),
                mode="markers",
                marker=dict(size=1.5, color="black"),
                name="EDMD 数据点",
            )
        )

    # 计算真实函数值（如果提供）
    if func_true is not None:
        zz_true = func_true(xx, yy)

        # 估计缩放因子（如果同时有EDMD和真实函数）
        scaling_factor = 1.0
        if func_edmd is not None and estimate_scaling:
            scaling_factor = _estimate_scaling_factor(
                zz_edmd.flatten(), zz_true.flatten()
            )

        # 添加真实函数表面
        data.append(
            go.Surface(
                z=scaling_factor * zz_true,
                x=xx,
                y=yy,
                showscale=True,
                opacity=0.5,
                colorbar=dict(lenmode="fraction", len=0.6, thickness=18),
                name="真实函数表面",
            )
        )

    # 创建图形
    fig = go.Figure(data=data)

    # 构建标题
    fig_title = title
    if true_eigenval is not None and edmd_eigenval is not None:
        fig_title = (
            f"TRUE vs EDMD; TRUE eig_val: {true_eigenval}; "
            f"EDMD eig_val: {edmd_eigenval}"
        )
    elif true_eigenval is not None:
        fig_title = f"{title}; TRUE eig_val: {true_eigenval}"
    elif edmd_eigenval is not None:
        fig_title = f"{title}; EDMD eig_val: {edmd_eigenval}"

    # 更新布局
    fig.update_layout(
        title=fig_title,
        autosize=True,
        width=1200,
        height=1200,
        margin=dict(l=0, r=0, b=15, t=30),
    )

    # 更新场景设置
    fig.update_layout(
        scene=dict(
            xaxis_title="x1", yaxis_title="x2", zaxis_title="函数值", aspectmode="cube"
        )
    )

    # 更新颜色轴
    fig.update_coloraxes(colorbar_xpad=0)

    # 保存HTML文件
    if save_html is not None:
        try:
            fig.write_html(save_html)
            print(f"✓ 图形已保存为HTML文件: {save_html}")
        except Exception as e:
            print(f"⚠ 保存HTML文件失败: {e}")

    # 显示图形
    if show_fig:
        try:
            fig.show()
        except Exception as e:
            print(f"⚠ 无法在浏览器中显示图形: {e}")
            print("提示: 可以使用save_html参数将图形保存为HTML文件")

    return fig


def _estimate_scaling_factor(data_edmd, data_true):
    """
    估计缩放因子以匹配EDMD和真实数据

    Args:
        data_edmd: EDMD数据（展平）
        data_true: 真实数据（展平）

    Returns:
        float: 缩放因子
    """
    # 使用最小二乘法估计缩放因子
    # 求解: min ||c * data_true - data_edmd||^2
    # 解: c = (data_true^T * data_edmd) / (data_true^T * data_true)

    numerator = np.dot(data_true, data_edmd)
    denominator = np.dot(data_true, data_true)

    if abs(denominator) < 1e-12:
        return 1.0

    return numerator / denominator


def visualize_eigenfunction_comparison_plotly(
    system,
    eigenvals_edmd,
    eigenfuncs_edmd,
    domain_x=(-2, 2),
    domain_y=(-2, 2),
    num_pts=100,
    show_fig=True,
    save_html_prefix=None,
):
    """
    使用plotly比较真实特征函数和EDMD特征函数

    Args:
        system: NL_EIG_System实例，包含真实特征函数
        eigenvals_edmd: EDMD特征值列表
        eigenfuncs_edmd: EDMD特征函数列表（callable函数）
        domain_x: x轴范围
        domain_y: y轴范围
        num_pts: 网格点数
        show_fig: 是否显示图形
        save_html_prefix: 如果提供，将为每个图形保存HTML文件（文件名为prefix_i.html）

    Returns:
        List[plotly.graph_objects.Figure]: 图形列表
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly库未安装。请安装: pip install plotly")

    # 获取真实特征值和特征函数
    true_eigenvals = system.get_eigenvalues()
    true_eigenfuncs = system.get_eigenfunctions()

    figures = []

    # 为每个特征函数创建比较图
    for i, (edmd_val, edmd_func) in enumerate(zip(eigenvals_edmd, eigenfuncs_edmd)):
        if i < len(true_eigenvals):
            # 创建真实函数的numpy函数
            true_func_symbolic = true_eigenfuncs[i]
            x_vars = sp.symbols("x:2")
            true_func_numpy = sp.lambdify(x_vars, true_func_symbolic, "numpy")

            # 准备保存文件名
            save_html_file = None
            if save_html_prefix is not None:
                save_html_file = f"{save_html_prefix}_{i+1}.html"

            # 创建图形
            fig = visualize_scalar_function_3d_plotly(
                func_true=true_func_numpy,
                func_edmd=edmd_func,
                domain_x=domain_x,
                domain_y=domain_y,
                num_pts=num_pts,
                title=f"特征函数 {i+1} 比较",
                true_eigenval=true_eigenvals[i],
                edmd_eigenval=edmd_val,
                show_fig=show_fig,
                save_html=save_html_file,
            )

            figures.append(fig)

    return figures
