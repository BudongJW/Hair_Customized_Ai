import * as ImagePicker from "expo-image-picker";
import React, { useMemo, useState } from "react";
import { Alert, Image, StyleSheet, Text, View } from "react-native";
import { api } from "../api/client";
import { Panel } from "../components/Panel";
import { PrimaryButton } from "../components/PrimaryButton";
import { StatusPill } from "../components/StatusPill";
import { TextField } from "../components/TextField";
import { colors, radius, spacing } from "../theme/tokens";

export function FaceProfileScreen({
  user,
  faceProfile,
  onBack,
  onProfileChanged,
  onProfileCompleted
}) {
  const [imageUri, setImageUri] = useState(null);
  const [fileName, setFileName] = useState("front-face.jpg");
  const [upload, setUpload] = useState(null);
  const [loading, setLoading] = useState(false);

  const profileReady = faceProfile?.status === "COMPLETED";
  const canCreateProfile = Boolean(user?.id && imageUri && fileName.trim());

  const profileLabel = useMemo(() => {
    if (!faceProfile) {
      return "READY";
    }
    return faceProfile.status;
  }, [faceProfile]);

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("사진 권한 필요", "얼굴 사진을 선택하려면 사진 접근 권한이 필요합니다.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [4, 5],
      quality: 0.72
    });

    if (result.canceled) {
      return;
    }

    const asset = result.assets[0];
    setImageUri(asset.uri);
    setFileName(asset.fileName || "front-face.jpg");
  };

  const createProfile = async () => {
    if (!canCreateProfile) {
      Alert.alert("사진 선택 필요", "먼저 내 얼굴 사진을 선택해 주세요.");
      return;
    }

    try {
      setLoading(true);
      const uploadResponse = await api.createPresignedUrl({
        fileName,
        contentType: "image/jpeg",
        purpose: "USER_FACE"
      });
      await api.uploadLocalImage(uploadResponse, imageUri, "image/jpeg");
      setUpload(uploadResponse);

      const profile = await api.createFaceProfile({
        userId: user.id,
        originalImageObjectKey: uploadResponse.objectKey
      });
      onProfileChanged(profile);
      Alert.alert("AI 분석 요청 완료", "Python AI worker가 S3 이미지를 분석하고 DB를 업데이트합니다.");
    } catch (error) {
      Alert.alert("얼굴 등록 실패", error.message);
    } finally {
      setLoading(false);
    }
  };

  const refreshProfile = async () => {
    if (!faceProfile?.id) {
      return;
    }

    try {
      setLoading(true);
      const profile = await api.getFaceProfile(faceProfile.id);
      onProfileChanged(profile);
      if (profile.status === "COMPLETED") {
        onProfileCompleted(profile);
      }
    } catch (error) {
      Alert.alert("상태 새로고침 실패", error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.headerRow}>
        <View style={styles.headerCopy}>
          <Text style={styles.title}>내 얼굴 등록</Text>
          <Text style={styles.subtitle}>{user?.displayName}님의 얼굴 프로필</Text>
        </View>
        <StatusPill status={profileLabel} />
      </View>

      <Panel>
        <View style={styles.preview}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.image} />
          ) : (
            <View style={styles.placeholder}>
              <View style={styles.faceOval} />
              <View style={styles.shoulderLine} />
              <Text style={styles.placeholderText}>정면 얼굴 사진</Text>
            </View>
          )}
        </View>

        <TextField
          label="파일명"
          value={fileName}
          onChangeText={setFileName}
          placeholder="front-face.jpg"
        />

        <View style={styles.buttonRow}>
          <PrimaryButton label="사진 선택" variant="secondary" onPress={pickImage} />
          <PrimaryButton
            label="얼굴 프로필 저장"
            onPress={createProfile}
            loading={loading}
            disabled={!canCreateProfile}
          />
        </View>
      </Panel>

      {faceProfile && (
        <Panel>
          <View style={styles.resultHeader}>
            <View style={styles.resultCopy}>
              <Text style={styles.sectionTitle}>저장된 얼굴 프로필</Text>
              <Text style={styles.objectKey}>{faceProfile.originalImageObjectKey}</Text>
              {faceProfile.baldCanvasObjectKey && (
                <Text style={styles.objectKey}>분석 결과: {faceProfile.baldCanvasObjectKey}</Text>
              )}
            </View>
            <StatusPill status={faceProfile.status} />
          </View>
          <PrimaryButton
            label={profileReady ? "분석 완료" : "AI 분석 상태 새로고침"}
            onPress={refreshProfile}
            loading={loading}
          />
        </Panel>
      )}

      {upload && (
        <Panel style={styles.compactPanel}>
          <Text style={styles.metaLabel}>UPLOAD OBJECT</Text>
          <Text style={styles.objectKey}>{upload.objectKey}</Text>
        </Panel>
      )}

      <PrimaryButton label="홈으로" variant="secondary" onPress={onBack} />
    </View>
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
    justifyContent: "space-between"
  },
  headerCopy: {
    flex: 1,
    gap: spacing.xs
  },
  title: {
    color: colors.ink,
    fontSize: 26,
    fontWeight: "900"
  },
  subtitle: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: "700"
  },
  preview: {
    backgroundColor: "#EEF2F6",
    borderRadius: radius.md,
    height: 280,
    overflow: "hidden"
  },
  image: {
    height: "100%",
    width: "100%"
  },
  placeholder: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center"
  },
  placeholderText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "800",
    marginTop: spacing.sm
  },
  faceOval: {
    backgroundColor: "#F3C9B8",
    borderColor: colors.coral,
    borderRadius: 54,
    borderWidth: 5,
    height: 108,
    width: 86
  },
  shoulderLine: {
    backgroundColor: colors.blue,
    borderTopLeftRadius: 52,
    borderTopRightRadius: 52,
    height: 42,
    marginTop: -5,
    width: 148
  },
  buttonRow: {
    flexDirection: "row",
    gap: spacing.sm
  },
  resultHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.md,
    justifyContent: "space-between"
  },
  resultCopy: {
    flex: 1,
    gap: spacing.xs
  },
  sectionTitle: {
    color: colors.ink,
    fontSize: 18,
    fontWeight: "900"
  },
  objectKey: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "700",
    lineHeight: 17
  },
  compactPanel: {
    gap: spacing.xs
  },
  metaLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "800"
  }
});
