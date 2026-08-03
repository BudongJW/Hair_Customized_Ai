import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { Panel } from "../components/Panel";
import { PrimaryButton } from "../components/PrimaryButton";
import { StatusPill } from "../components/StatusPill";
import { colors, radius, spacing } from "../theme/tokens";

export function HomeScreen({
  user,
  loadingUserData = false,
  faceProfile,
  fittingJob,
  fittingJobCount = 0,
  onLogout,
  onOpenFaceProfile,
  onOpenHairFitting,
  onOpenHistory
}) {
  return (
    <View style={styles.wrap}>
      <View style={styles.headerRow}>
        <View style={styles.headerCopy}>
          <Text style={styles.greeting}>{user.displayName}님</Text>
          <Text style={styles.subtitle}>오늘 시도해볼 헤어 피팅을 선택하세요.</Text>
        </View>
        <PrimaryButton label="로그아웃" variant="secondary" onPress={onLogout} />
      </View>

      <View style={styles.statusRow}>
        <Panel style={styles.statusPanel}>
          <Text style={styles.statusLabel}>얼굴 프로필</Text>
          <StatusPill status={faceProfile?.status || "READY"} />
        </Panel>
        <Panel style={styles.statusPanel}>
          <Text style={styles.statusLabel}>최근 피팅</Text>
          <StatusPill status={fittingJob?.status || "READY"} />
        </Panel>
      </View>

      <Text style={styles.restoreText}>
        {loadingUserData ? "저장된 데이터를 불러오는 중입니다." : `저장된 피팅 기록 ${fittingJobCount}개`}
      </Text>

      <View style={styles.menuList}>
        <MenuCard
          title="내 얼굴 등록"
          body="정면 사진을 등록하고 AI 전처리용 얼굴 프로필을 만듭니다."
          accent={colors.coral}
          onPress={onOpenFaceProfile}
        />
        <MenuCard
          title="합성하고 싶은 헤어스타일 모델 등록"
          body="헤어스타일이 잘 보이는 모델 사진을 등록하고 피팅 작업을 생성합니다."
          accent={colors.teal}
          onPress={onOpenHairFitting}
        />
        <MenuCard
          title="예전 결과물 보기"
          body="완료된 피팅 결과와 생성된 결과 이미지 경로를 확인합니다."
          accent={colors.gold}
          onPress={onOpenHistory}
        />
      </View>
    </View>
  );
}

function MenuCard({ title, body, accent, onPress }) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.menuCard, pressed && styles.pressed]}>
      <View style={[styles.accentBar, { backgroundColor: accent }]} />
      <View style={styles.menuCopy}>
        <Text style={styles.menuTitle}>{title}</Text>
        <Text style={styles.menuBody}>{body}</Text>
      </View>
      <Text style={styles.arrow}>{">"}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.md
  },
  headerRow: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.md,
    justifyContent: "space-between",
    paddingTop: spacing.sm
  },
  headerCopy: {
    flex: 1,
    gap: spacing.xs
  },
  greeting: {
    color: colors.ink,
    fontSize: 28,
    fontWeight: "900"
  },
  subtitle: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "700",
    lineHeight: 20
  },
  statusRow: {
    flexDirection: "row",
    gap: spacing.sm
  },
  statusPanel: {
    flex: 1,
    gap: spacing.sm
  },
  statusLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "900"
  },
  restoreText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "800",
    textAlign: "center"
  },
  menuList: {
    gap: spacing.md
  },
  menuCard: {
    alignItems: "center",
    backgroundColor: colors.panel,
    borderColor: colors.line,
    borderRadius: radius.md,
    borderWidth: 1,
    flexDirection: "row",
    minHeight: 112,
    overflow: "hidden"
  },
  pressed: {
    opacity: 0.78
  },
  accentBar: {
    alignSelf: "stretch",
    width: 8
  },
  menuCopy: {
    flex: 1,
    gap: spacing.xs,
    padding: spacing.md
  },
  menuTitle: {
    color: colors.ink,
    fontSize: 18,
    fontWeight: "900",
    lineHeight: 24
  },
  menuBody: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "600",
    lineHeight: 19
  },
  arrow: {
    color: colors.muted,
    fontSize: 28,
    fontWeight: "700",
    paddingRight: spacing.md
  }
});
