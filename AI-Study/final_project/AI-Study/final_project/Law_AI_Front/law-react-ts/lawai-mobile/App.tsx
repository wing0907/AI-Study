import { useState } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  Pressable,
  Modal,
  StyleSheet,
  Platform,
} from 'react-native';

export default function App() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');

  return (
    <SafeAreaView style={s.root}>
      {/* Topbar */}
      <View style={s.topbar}>
        <Text style={s.brand}>LawAI</Text>
        <View style={{ flexDirection: 'row', gap: 8 }}>
          <Pressable style={[s.btn, s.ghost]}>
            <Text style={s.btnGhostText}>로그인</Text>
          </Pressable>
          <Pressable style={[s.btn, s.primary]}>
            <Text style={s.btnPrimaryText}>무료 가입</Text>
          </Pressable>
        </View>
      </View>

      {/* Card */}
      <View style={s.card}>
        <Text style={s.title}>지능형 리서치</Text>
        <Text style={s.caption}>질문을 입력하면 관련 법률 문서 일부를 검색해 보여줍니다.</Text>

        <View style={s.empty}>
          <Text style={s.emptyText}>아직 메시지가 없습니다.</Text>
        </View>
      </View>

      {/* Bottom input */}
      <View style={s.dock}>
        <Pressable style={s.plus} onPress={() => setOpen(true)}>
          <Text style={s.plusT}>＋</Text>
        </Pressable>
        <TextInput
          placeholder="무엇을 도와드릴까요?"
          placeholderTextColor="#667187"
          value={text}
          onChangeText={setText}
          style={s.input}
          onSubmitEditing={() => {
            if (!text.trim()) return;
            setText('');
            setOpen(false);
          }}
        />
        <Pressable
          style={s.send}
          onPress={() => {
            if (!text.trim()) return;
            setText('');
            setOpen(false);
          }}
        >
          <Text style={{ color: '#fff', fontWeight: '700' }}>전송</Text>
        </Pressable>
      </View>

      {/* Attach popover (모달) */}
      <Modal transparent visible={open} animationType="fade" onRequestClose={() => setOpen(false)}>
        <Pressable style={s.backdrop} onPress={() => setOpen(false)}>
          <View style={s.sheet} pointerEvents="box-none">
            <View style={s.sheetInner}>
              <Pressable style={s.chip}><Text>🖼 이미지</Text></Pressable>
              <Pressable style={s.chip}><Text>🎙 음성</Text></Pressable>
              <Pressable style={s.chip}><Text>📎 파일</Text></Pressable>
            </View>
          </View>
        </Pressable>
      </Modal>
    </SafeAreaView>
  );
}

const BLUE900 = '#0B2447';
const BLUE700 = '#19376D';
const BLUE500 = '#1D4ED8';

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: BLUE900 },
  topbar: {
    height: 56,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderColor: BLUE700,
  },
  brand: { color: '#fff', fontWeight: '800', fontSize: 18, letterSpacing: 0.2 },
  btn: {
    height: 36,
    paddingHorizontal: 12,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  ghost: { borderColor: BLUE700, backgroundColor: 'transparent' },
  btnGhostText: { color: '#fff' },
  primary: { backgroundColor: BLUE500, borderColor: 'transparent' },
  btnPrimaryText: { color: '#fff', fontWeight: '700' },

  card: {
    margin: 16,
    backgroundColor: '#fff',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: BLUE700,
    padding: 16,
    shadowColor: '#000',
    shadowOpacity: 0.25,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    ...Platform.select({
      android: { elevation: 6 },
    }),
  },
  title: { color: '#0f172a', fontWeight: '800', fontSize: 18, marginBottom: 6 },
  caption: { color: '#1D4ED8' },
  empty: {
    marginTop: 10,
    borderRadius: 12,
    backgroundColor: '#F6F8FF',
    borderWidth: 1,
    borderColor: BLUE700,
    padding: 12,
  },
  emptyText: { color: '#0f172a' },

  dock: {
    position: 'absolute',
    left: 12,
    right: 12,
    bottom: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  plus: {
    width: 40, height: 40, borderRadius: 10,
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: BLUE700, backgroundColor: '#fff',
  },
  plusT: { color: '#0f172a', fontSize: 18 },
  input: {
    flex: 1, height: 40, backgroundColor: '#fff', borderRadius: 10,
    paddingHorizontal: 10, borderWidth: 1, borderColor: BLUE700, color: '#0f172a',
  },
  send: {
    paddingHorizontal: 14, height: 40, borderRadius: 10,
    alignItems: 'center', justifyContent: 'center', backgroundColor: BLUE500,
  },

  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.35)', justifyContent: 'flex-end' },
  sheet: { width: '100%', paddingHorizontal: 12, paddingBottom: 20 },
  sheetInner: {
    flexDirection: 'row', gap: 8, backgroundColor: '#fff',
    padding: 10, borderRadius: 12, borderWidth: 1, borderColor: BLUE700,
  },
  chip: {
    paddingVertical: 6, paddingHorizontal: 10,
    borderWidth: 1, borderColor: BLUE700, borderRadius: 999, backgroundColor: '#F6F8FC',
  },
});
