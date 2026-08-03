import React, { useState } from "react";
import { Alert, Pressable, StyleSheet, Text, View } from "react-native";
import { api, API_BASE_URL } from "../api/client";
import { Panel } from "../components/Panel";
import { PrimaryButton } from "../components/PrimaryButton";
import { TextField } from "../components/TextField";
import { colors, radius, spacing } from "../theme/tokens";

export function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const isRegister = mode === "register";

  const submit = async () => {
    try {
      setLoading(true);
      const payload = isRegister
        ? { email, displayName, password }
        : { email, password };
      const user = isRegister
        ? await api.registerWithPassword(payload)
        : await api.loginWithPassword(payload);
      onAuthenticated(user);
    } catch (error) {
      Alert.alert(isRegister ? "회원가입 실패" : "로그인 실패", error.message);
    } finally {
      setLoading(false);
    }
  };

  const showGoogleNotice = () => {
    Alert.alert("Google 로그인", "OAuth2 클라이언트 설정 후 연결할 예정입니다.");
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.brandBlock}>
        <Text style={styles.brand}>Hair Fit Studio</Text>
        <Text style={styles.tagline}>내 얼굴에 어울리는 헤어스타일을 미리 확인하세요.</Text>
      </View>

      <Panel>
        <PrimaryButton label="Google로 계속하기" variant="secondary" onPress={showGoogleNotice} />

        <View style={styles.dividerRow}>
          <View style={styles.divider} />
          <Text style={styles.dividerText}>또는 이메일로 시작</Text>
          <View style={styles.divider} />
        </View>

        <View style={styles.segmented}>
          <Pressable
            onPress={() => setMode("login")}
            style={[styles.segment, !isRegister && styles.activeSegment]}
          >
            <Text style={[styles.segmentText, !isRegister && styles.activeSegmentText]}>로그인</Text>
          </Pressable>
          <Pressable
            onPress={() => setMode("register")}
            style={[styles.segment, isRegister && styles.activeSegment]}
          >
            <Text style={[styles.segmentText, isRegister && styles.activeSegmentText]}>회원가입</Text>
          </Pressable>
        </View>

        <TextField
          label="이메일"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          placeholder="you@example.com"
        />

        {isRegister && (
          <TextField
            label="이름"
            value={displayName}
            onChangeText={setDisplayName}
            placeholder="사용자 이름"
          />
        )}

        <TextField
          label="비밀번호"
          value={password}
          onChangeText={setPassword}
          placeholder="6자 이상"
          secureTextEntry
        />

        <PrimaryButton
          label={isRegister ? "계정 만들기" : "로그인"}
          onPress={submit}
          loading={loading}
          disabled={!email || password.length < 6 || (isRegister && !displayName)}
        />
      </Panel>

      <Panel style={styles.apiPanel}>
        <Text style={styles.apiLabel}>API 서버</Text>
        <Text style={styles.apiText}>{API_BASE_URL}</Text>
      </Panel>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.md
  },
  brandBlock: {
    gap: spacing.sm,
    paddingBottom: spacing.sm,
    paddingTop: spacing.md
  },
  brand: {
    color: colors.ink,
    fontSize: 34,
    fontWeight: "900"
  },
  tagline: {
    color: colors.muted,
    fontSize: 15,
    fontWeight: "700",
    lineHeight: 22
  },
  dividerRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: spacing.sm
  },
  divider: {
    backgroundColor: colors.line,
    flex: 1,
    height: 1
  },
  dividerText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "800"
  },
  segmented: {
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    flexDirection: "row",
    padding: 4
  },
  segment: {
    alignItems: "center",
    borderRadius: radius.sm,
    flex: 1,
    paddingVertical: spacing.sm
  },
  activeSegment: {
    backgroundColor: colors.panel
  },
  segmentText: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "800"
  },
  activeSegmentText: {
    color: colors.ink
  },
  apiPanel: {
    gap: spacing.xs
  },
  apiLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "900"
  },
  apiText: {
    color: colors.ink,
    fontSize: 13,
    fontWeight: "700"
  }
});
