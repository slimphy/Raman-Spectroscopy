import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch)
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class RamanSRNetV21(nn.Module):
    def __init__(self, num_features=64, num_blocks=6):
        super().__init__()
        self.head = nn.Sequential(nn.Conv1d(1, num_features, 9, padding=4), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv1d(num_features, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 1, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.tail(self.body(self.head(x)))


class RamanMLProcessor:
    """스펙트럼 ML 처리기 (UI에서 이 클래스만 생성하여 호출)"""

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            # 상태 사전을 불러와서 동적으로 구조(features, blocks) 파악 후 모델 생성
            state_dict = torch.load(model_path, map_location=self.device)
            num_features = state_dict.get('head.0.weight', torch.zeros((64,))).shape[0]
            num_blocks = len([k for k in state_dict.keys() if 'body' in k and 'net.0.weight' in k])

            self.model = RamanSRNetV21(num_features=num_features, num_blocks=num_blocks).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"ML Model successfully loaded on {self.device}")
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            self.model = None

    def enhance_spectrum(self, spectrum: np.ndarray, noise_cutoff_ratio: float = 0.0015) -> np.ndarray:
        """1D 스펙트럼 배열을 받아 ML 향상된 배열을 반환"""
        if self.model is None:
            return spectrum

        original_max = np.max(spectrum)
        if original_max < 1e-12:
            return np.zeros_like(spectrum)

        # 정규화
        norm_spectrum = spectrum / original_max
        input_tensor = torch.tensor(norm_spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # 추론
        with torch.no_grad():
            out_tensor = self.model(input_tensor)

        out_spectrum = out_tensor.squeeze().cpu().numpy()

        # 스케일 복원 및 노이즈 컷오프
        enhanced_spectrum = out_spectrum * original_max
        enhanced_spectrum[enhanced_spectrum < (np.max(enhanced_spectrum) * noise_cutoff_ratio)] = 0

        return enhanced_spectrum