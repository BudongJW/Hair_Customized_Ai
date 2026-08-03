# Hair Fit Studio Frontend

Expo 기반 React Native 초기 화면입니다.

## 실행

```powershell
cd C:\Users\aaron\Project\Hair_Customized_Ai\frontend
node .\node_modules\expo\bin\cli start -c
```

## 현재 흐름

1. Google 로그인 버튼은 OAuth2 설정 전 UI 목업입니다.
2. 이메일/비밀번호 로그인과 회원가입은 백엔드 API에 연결되어 있습니다.
3. 로그인 후 홈에서 `내 얼굴 등록`, `합성하고 싶은 헤어스타일 모델 등록`, `예전 결과물 보기`로 이동합니다.
4. 얼굴/헤어 사진 저장 후 Python AI worker가 처리한 결과는 `상태 새로고침`으로 DB에서 다시 불러옵니다.

## API 주소

실제 휴대폰의 Expo Go에서 테스트할 때 `localhost`는 PC가 아니라 휴대폰 자신을 의미합니다.
그래서 `.env`에는 PC의 같은 Wi-Fi IPv4 주소를 넣어야 합니다.

```text
EXPO_PUBLIC_API_BASE_URL=http://192.168.0.12:8080
```

PC의 IP가 바뀌면 PowerShell에서 `ipconfig`로 `Wireless LAN adapter Wi-Fi`의 IPv4 주소를 확인한 뒤 `.env`를 수정하세요.
