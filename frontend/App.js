import React from "react";
import { ActivityIndicator, Alert, StyleSheet, Text, View } from "react-native";
import { api } from "./src/api/client";
import { Screen } from "./src/components/Screen";
import { AuthScreen } from "./src/screens/AuthScreen";
import { HomeScreen } from "./src/screens/HomeScreen";
import { FaceProfileScreen } from "./src/screens/FaceProfileScreen";
import { HairFittingScreen } from "./src/screens/HairFittingScreen";
import { HistoryScreen } from "./src/screens/HistoryScreen";
import { ResultScreen } from "./src/screens/ResultScreen";
import { colors, spacing } from "./src/theme/tokens";

function choosePrimaryFaceProfile(profiles) {
  return profiles.find((profile) => profile.status === "COMPLETED") || profiles[0] || null;
}

function chooseCompletedFaceProfile(profiles) {
  return profiles.find((profile) => profile.status === "COMPLETED") || null;
}

function replaceById(items, nextItem) {
  const exists = items.some((item) => item.id === nextItem.id);
  if (!exists) {
    return [nextItem, ...items];
  }
  return items.map((item) => (item.id === nextItem.id ? nextItem : item));
}

const ACTIVE_FITTING_STATUSES = new Set(["PENDING", "PROCESSING"]);
const FINISHED_FITTING_STATUSES = new Set(["PREPARED", "COMPLETED", "FAILED", "REJECTED"]);

export default function App() {
  const [view, setView] = React.useState("auth");
  const [user, setUser] = React.useState(null);
  const [loadingUserData, setLoadingUserData] = React.useState(false);
  const [faceProfiles, setFaceProfiles] = React.useState([]);
  const [faceProfile, setFaceProfile] = React.useState(null);
  const [fittingJobs, setFittingJobs] = React.useState([]);
  const [fittingJob, setFittingJob] = React.useState(null);

  React.useEffect(() => {
    if (!fittingJob?.id || !ACTIVE_FITTING_STATUSES.has(fittingJob.status)) {
      return undefined;
    }

    let cancelled = false;
    let timerId;

    const pollFittingJob = async () => {
      try {
        const latestJob = await api.getFittingJob(fittingJob.id);
        if (cancelled) {
          return;
        }
        setFittingJobs((items) => replaceById(items, latestJob));
        setFittingJob(latestJob);
        if (FINISHED_FITTING_STATUSES.has(latestJob.status)) {
          setView((currentView) => currentView === "hair" ? "result" : currentView);
          return;
        }
      } catch {
        // Transient mobile-network failures should not force a logout or stop polling.
      }
      if (!cancelled) {
        timerId = setTimeout(pollFittingJob, 2000);
      }
    };

    timerId = setTimeout(pollFittingJob, 800);
    return () => {
      cancelled = true;
      clearTimeout(timerId);
    };
  }, [fittingJob?.id, fittingJob?.status]);

  const restoreUserData = async (authenticatedUser) => {
    setUser(authenticatedUser);
    setView("home");
    setLoadingUserData(true);

    try {
      const [profiles, jobs] = await Promise.all([
        api.listFaceProfiles(authenticatedUser.id),
        api.listFittingJobs(authenticatedUser.id)
      ]);

      setFaceProfiles(profiles);
      setFaceProfile(choosePrimaryFaceProfile(profiles));
      setFittingJobs(jobs);
      setFittingJob(jobs[0] || null);
    } catch (error) {
      Alert.alert("데이터 불러오기 실패", error.message);
      setFaceProfiles([]);
      setFaceProfile(null);
      setFittingJobs([]);
      setFittingJob(null);
    } finally {
      setLoadingUserData(false);
    }
  };

  const logout = () => {
    setUser(null);
    setLoadingUserData(false);
    setFaceProfiles([]);
    setFaceProfile(null);
    setFittingJobs([]);
    setFittingJob(null);
    setView("auth");
  };

  const updateFaceProfile = (profile) => {
    setFaceProfiles((items) => replaceById(items, profile));
    setFaceProfile(profile);
  };

  const updateFittingJob = (job) => {
    setFittingJobs((items) => replaceById(items, job));
    setFittingJob(job);
  };

  const openHairFitting = async () => {
    if (loadingUserData) {
      return;
    }

    try {
      setLoadingUserData(true);
      const profiles = await api.listFaceProfiles(user.id);
      const completedProfile = chooseCompletedFaceProfile(profiles);

      setFaceProfiles(profiles);
      setFaceProfile(completedProfile || choosePrimaryFaceProfile(profiles));

      if (!completedProfile) {
        const analysisInProgress = profiles.some((profile) =>
          ["PENDING", "PROCESSING"].includes(profile.status)
        );
        Alert.alert(
          analysisInProgress ? "얼굴 분석 진행 중" : "얼굴 분석 필요",
          analysisInProgress
            ? "등록한 얼굴의 AI 분석이 아직 끝나지 않았습니다. 얼굴 등록 화면에서 분석 상태를 새로고침해 주세요."
            : "먼저 내 얼굴을 등록하고 AI 분석을 완료해 주세요."
        );
        setView("face");
        return;
      }

      setView("hair");
    } catch (error) {
      Alert.alert("얼굴 프로필 확인 실패", error.message);
    } finally {
      setLoadingUserData(false);
    }
  };

  if (!user) {
    return (
      <Screen>
        <AuthScreen onAuthenticated={restoreUserData} />
      </Screen>
    );
  }

  return (
    <Screen>
      {view === "home" && (
        <HomeScreen
          user={user}
          loadingUserData={loadingUserData}
          faceProfile={faceProfile}
          fittingJob={fittingJob}
          fittingJobCount={fittingJobs.length}
          onLogout={logout}
          onOpenFaceProfile={() => setView("face")}
          onOpenHairFitting={openHairFitting}
          onOpenHistory={() => setView("history")}
        />
      )}

      {view === "face" && (
        <FaceProfileScreen
          user={user}
          faceProfile={faceProfile}
          onBack={() => setView("home")}
          onProfileChanged={updateFaceProfile}
          onProfileCompleted={(profile) => {
            updateFaceProfile(profile);
            Alert.alert("AI 분석 완료", "얼굴 분석 결과가 DB에 저장되었습니다.");
          }}
        />
      )}

      {view === "hair" && (
        <HairFittingScreen
          user={user}
          faceProfile={faceProfile}
          fittingJob={fittingJob}
          onBack={() => setView("home")}
          onFittingJobChanged={updateFittingJob}
          onFittingCompleted={(job) => {
            updateFittingJob(job);
            setView("result");
          }}
        />
      )}

      {view === "history" && (
        <HistoryScreen
          user={user}
          fittingJobs={fittingJobs}
          onBack={() => setView("home")}
          onOpenResult={(job) => {
            const matchingProfile = faceProfiles.find((profile) => profile.id === job.profileId);
            if (matchingProfile) {
              setFaceProfile(matchingProfile);
            }
            setFittingJob(job);
            setView("result");
          }}
        />
      )}

      {view === "result" && (
        <ResultScreen
          user={user}
          faceProfile={faceProfile}
          fittingJob={fittingJob}
          onTryAnother={() => {
            setFittingJob(null);
            setView("hair");
          }}
          onReset={() => setView("home")}
        />
      )}

      {loadingUserData && (
        <View style={styles.loadingBox}>
          <ActivityIndicator />
          <Text style={styles.loadingText}>저장된 작업을 불러오는 중입니다.</Text>
        </View>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  loadingBox: {
    alignItems: "center",
    gap: spacing.sm,
    paddingVertical: spacing.sm
  },
  loadingText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: "700"
  }
});
